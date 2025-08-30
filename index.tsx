import React, { useState, useRef, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { GoogleGenAI, Chat } from "@google/genai";

const API_KEY = process.env.API_KEY;

// A concise system instruction based on the user's detailed prompt.
const SYSTEM_INSTRUCTION = `You are a world-class personal assistant AI. You are designed to help users with a wide range of tasks including:
- Academics: Summarize notes, generate study guides, solve problems, and act as a tutor.
- Productivity & Organization: Help prioritize tasks, build routines, and manage schedules.
- Research & Automation: Find and synthesize information, automate repetitive tasks, and provide proactive suggestions.
- Engineering & Coding: Write, debug, and explain code. Analyze schematics and assist with maker projects (Arduino, 3D printing). When you provide runnable code (like JavaScript or HTML), make sure it is self-contained and executable in a browser environment without external dependencies.
- Image Generation: You have the ability to generate images. If a user asks you to create, draw, or generate an image, you must respond *only* with a JSON object in the following format: {"image_prompt": "a detailed description of the image to generate"}. Do not include any other text, explanations, or markdown formatting in your response. Your entire response must be the raw JSON object.
Maintain a helpful, adaptive, and proactive personality. Remember user preferences and context to provide personalized support.`;

// @ts-ignore
const SpeechRecognition =
  (window as any).SpeechRecognition ||
  (window as any).webkitSpeechRecognition;

interface TextPart {
  type: "text";
  content: string;
}
interface CodePart {
  type: "code";
  content: string;
  language?: string;
}
interface ImagePart {
  type: "image";
  imageUrl: string;
  prompt: string;
}

type MessagePart = TextPart | CodePart | ImagePart;

interface Message {
  role: "user" | "ai";
  parts: MessagePart[];
}

const getApiErrorMessage = (error: any): string => {
  if (!navigator.onLine) {
    return "It seems you're offline. Please check your internet connection.";
  }

  const message = (error?.message || String(error)).toLowerCase();

  if (
    message.includes("api key not valid") ||
    message.includes("permission denied")
  ) {
    return "There seems to be an issue with the API key. Please verify it is correct and has the required permissions.";
  }

  if (message.includes("quota")) {
    return "The API quota has been exceeded. Please check your usage limits.";
  }

  if (
    message.includes("500") ||
    message.includes("503") ||
    message.includes("server error")
  ) {
    return "The AI service is currently experiencing issues. Please try again in a few moments.";
  }

  if (message.includes("fetch") || message.includes("network")) {
    return "A network error occurred while trying to reach the AI service. Please check your connection.";
  }

  return `An unexpected error occurred. Details: ${
    error?.message || "No additional information available."
  }`;
};

const parseResponse = (text: string): MessagePart[] => {
  const parts: MessagePart[] = [];
  const regex = /```(\w*)\n([\s\S]+?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({
        type: "text",
        content: text.substring(lastIndex, match.index),
      });
    }
    parts.push({
      type: "code",
      language: match[1].toLowerCase() || "plaintext",
      content: match[2].trim(),
    });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push({ type: "text", content: text.substring(lastIndex) });
  }

  return parts.length > 0 ? parts : [{ type: "text", content: text }];
};

const CodeBlock = ({ code, language }: { code: string; language?: string }) => {
  const [output, setOutput] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleRunCode = () => {
    setOutput([]);
    setError(null);
    const logs: string[] = [];
    const originalConsoleLog = console.log;
    console.log = (...args) => {
      logs.push(
        args
          .map((arg) => {
            try {
              return typeof arg === "object"
                ? JSON.stringify(arg, null, 2)
                : String(arg);
            } catch (e) {
              return "Could not stringify object";
            }
          })
          .join(" ")
      );
      originalConsoleLog.apply(console, args);
    };
    try {
      const result = new Function(code)();
      if (result !== undefined) {
        logs.push(`=> ${JSON.stringify(result, null, 2)}`);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      console.log = originalConsoleLog;
      setOutput(logs);
    }
  };

  const isRunnable = language === "javascript" || language === "js";

  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language}</span>
        {isRunnable && (
          <button onClick={handleRunCode} className="run-button">
            Run Code
          </button>
        )}
      </div>
      <pre>
        <code>{code}</code>
      </pre>
      {(output.length > 0 || error) && (
        <div className="code-output">
          <h4>Output</h4>
          <pre>
            {output.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            {error && <div className="error-output">{error}</div>}
          </pre>
        </div>
      )}
    </div>
  );
};

const HtmlBlock = ({ code }: { code: string }) => {
  const [showPreview, setShowPreview] = useState(false);
  return (
    <div className="code-block">
      <div className="code-header">
        <span>html</span>
        <div className="view-toggle">
          <button
            onClick={() => setShowPreview(false)}
            className={!showPreview ? "active" : ""}
            aria-pressed={!showPreview}
          >
            Code
          </button>
          <button
            onClick={() => setShowPreview(true)}
            className={showPreview ? "active" : ""}
            aria-pressed={showPreview}
          >
            Preview
          </button>
        </div>
      </div>
      {showPreview ? (
        <div className="html-preview">
          <iframe srcDoc={code} title="HTML Preview" sandbox="allow-scripts" />
        </div>
      ) : (
        <pre>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
};

const ImageBlock = ({
  imageUrl,
  prompt,
}: {
  imageUrl: string;
  prompt: string;
}) => {
  return (
    <div className="image-block">
      <div className="image-header" title={prompt}>
        <span>{prompt}</span>
      </div>
      <img src={imageUrl} alt={prompt} />
    </div>
  );
};

const App = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("Processing...");
  const [chat, setChat] = useState<Chat | null>(null);
  const [ai, setAi] = useState<GoogleGenAI | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isTtsEnabled, setIsTtsEnabled] = useState(false);
  const recognitionRef = useRef<any>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const isSpeechSupported = !!SpeechRecognition;
  const isTtsSupported = "speechSynthesis" in window;

  useEffect(() => {
    // Clean up speech synthesis on component unmount
    return () => {
      if (isTtsSupported) {
        window.speechSynthesis.cancel();
      }
    };
  }, [isTtsSupported]);

  useEffect(() => {
    async function initializeChat() {
      if (!API_KEY) {
        setMessages([
          {
            role: "ai",
            parts: [
              {
                type: "text",
                content:
                  "Configuration Error: The API key is missing. Please ensure it is set up correctly in your environment.",
              },
            ],
          },
        ]);
        return;
      }
      try {
        const genAI = new GoogleGenAI({ apiKey: API_KEY });
        setAi(genAI);
        const chatSession = genAI.chats.create({
          model: "gemini-2.5-flash",
          config: { systemInstruction: SYSTEM_INSTRUCTION },
        });
        setChat(chatSession);
      } catch (error) {
        console.error("Failed to initialize AI chat:", error);
        const friendlyMessage = getApiErrorMessage(error);
        setMessages([
          {
            role: "ai",
            parts: [
              {
                type: "text",
                content: `Error: Could not initialize the AI. ${friendlyMessage}`,
              },
            ],
          },
        ]);
      }
    }
    initializeChat();

    if (isSpeechSupported) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognition.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0])
          .map((result) => result.transcript)
          .join("");
        setInput(transcript);
      };
      recognition.onerror = (event: any) =>
        console.error("Speech recognition error:", event.error);
      recognition.onend = () => setIsListening(false);
      recognitionRef.current = recognition;
    }
  }, [isSpeechSupported]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  useEffect(() => {
    if (!isTtsEnabled || !isTtsSupported || messages.length === 0) return;

    const lastMessage = messages[messages.length - 1];
    if (lastMessage.role === "ai") {
      const textToSpeak = lastMessage.parts
        .filter((part) => part.type === "text" && part.content.trim())
        .map((part) => (part as TextPart).content)
        .join(" ");

      if (textToSpeak) {
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        window.speechSynthesis.speak(utterance);
      }
    }
  }, [messages, isTtsEnabled, isTtsSupported]);

  const handleImageGeneration = async (prompt: string) => {
    if (!ai) return;
    setLoadingText(`🎨 Generating image of "${prompt}"...`);
    try {
      const response = await ai.models.generateImages({
        model: "imagen-4.0-generate-001",
        prompt,
        config: { numberOfImages: 1, outputMimeType: "image/jpeg" },
      });
      const generatedImages = response.generatedImages;
      const base64ImageBytes =
        generatedImages &&
        generatedImages[0] &&
        generatedImages[0].image &&
        generatedImages[0].image.imageBytes
          ? generatedImages[0].image.imageBytes
          : undefined;
      if (!base64ImageBytes) throw new Error("No image data received.");
      const imageUrl = `data:image/jpeg;base64,${base64ImageBytes}`;
      const imageMessage: Message = {
        role: "ai",
        parts: [{ type: "image", imageUrl, prompt }],
      };
      setMessages((prev) => [...prev, imageMessage]);
    } catch (error) {
      console.error("Image generation failed:", error);
      const friendlyMessage = getApiErrorMessage(error);
      const errorMessage: Message = {
        role: "ai",
        parts: [
          {
            type: "text",
            content: `Sorry, I couldn't create the image. ${friendlyMessage}`,
          },
        ],
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !chat) return;
    if (isListening) recognitionRef.current?.stop();
    if (isTtsSupported) window.speechSynthesis.cancel();

    const userMessage: Message = {
      role: "user",
      parts: [{ type: "text", content: input }],
    };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);
    setLoadingText("Processing...");

    try {
      const response = await chat.sendMessage({ message: currentInput });
      const responseText = (response.text ?? "").trim();
      let isJson = false;
      try {
        const potentialJson = JSON.parse(responseText);
        if (
          potentialJson.image_prompt &&
          typeof potentialJson.image_prompt === "string"
        ) {
          isJson = true;
          await handleImageGeneration(potentialJson.image_prompt);
        }
      } catch (e) {
        /* Not a JSON response */
      }

      if (!isJson) {
        const aiParts = parseResponse(responseText);
        const aiMessage: Message = { role: "ai", parts: aiParts };
        setMessages((prev) => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const friendlyMessage = getApiErrorMessage(error);
      const errorMessage: Message = {
        role: "ai",
        parts: [
          {
            type: "text",
            content: `Sorry, something went wrong. ${friendlyMessage}`,
          },
        ],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleListen = () => {
    if (!isSpeechSupported) return;
    if (isTtsSupported) window.speechSynthesis.cancel();
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      setInput("");
      recognitionRef.current?.start();
      setIsListening(true);
    }
  };

  const handleTtsToggle = () => {
    if (!isTtsSupported) return;
    if (isTtsEnabled) {
      window.speechSynthesis.cancel();
    }
    setIsTtsEnabled((prev) => !prev);
  };

  return (
    <>
      <header>
        <h1>AXIS</h1>
        {isTtsSupported && (
          <button
            onClick={handleTtsToggle}
            className={`tts-button ${isTtsEnabled ? "active" : ""}`}
            aria-label={
              isTtsEnabled ? "Disable text-to-speech" : "Enable text-to-speech"
            }
          >
            {isTtsEnabled ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
              </svg>
            )}
          </button>
        )}
      </header>
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((msg, msgIndex) => (
          <div key={msgIndex} className={`chat-message ${msg.role}`}>
            <span className="sender">
              {msg.role === "user" ? "You" : "AXIS"}
            </span>
            {msg.parts.map((part, partIndex) => {
              switch (part.type) {
                case "code":
                  return part.language === "html" ? (
                    <HtmlBlock key={partIndex} code={part.content} />
                  ) : (
                    <CodeBlock
                      key={partIndex}
                      code={part.content}
                      language={part.language}
                    />
                  );
                case "image":
                  return (
                    <ImageBlock
                      key={partIndex}
                      imageUrl={part.imageUrl}
                      prompt={part.prompt}
                    />
                  );
                case "text":
                  return part.content.trim() ? (
                    <div key={partIndex} className="bubble">
                      {part.content}
                    </div>
                  ) : null;
                default:
                  return null;
              }
            })}
          </div>
        ))}
        {isLoading && (
          <div className="chat-message ai">
            <span className="sender">AXIS</span>
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>{loadingText}</span>
            </div>
          </div>
        )}
      </div>
      <form className="chat-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter command or hold mic to speak..."
            aria-label="Chat input"
            disabled={isLoading || !chat}
          />
          {isSpeechSupported && (
            <button
              type="button"
              onClick={handleListen}
              className={`mic-button ${isListening ? "listening" : ""}`}
              aria-label={isListening ? "Stop listening" : "Start listening"}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z" />
              </svg>
            </button>
          )}
        </div>
        <button
          type="submit"
          disabled={isLoading || !input.trim() || !chat}
          aria-label="Send message"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </form>
    </>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(<App />);
