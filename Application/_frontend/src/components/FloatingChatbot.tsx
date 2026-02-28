import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, Sparkles, MessageCircle, Loader2 } from "lucide-react";
import chachiaPng from "@/assets/chachia.png";
import TunisianPattern from "./TunisianPattern";
import {
  sessionStart,
  getSession,
  choucheneChat,
  qaChat,
  type ChouchanePhase,
  type ChouchaneResponse,
} from "@/services/chouchaneApi";
import { useChouchaneSession } from "@/context/ChouchaneSessionContext";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const quickSuggestions = [
  "I love culture and history",
  "Beach and relaxation",
  "Adventure in the desert",
  "Food and local experiences",
];

function appendAssistant(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  content: string
) {
  setMessages((prev) => [
    ...prev,
    { id: `a-${Date.now()}`, role: "assistant", content },
  ]);
}

function appendError(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  message: string
) {
  setMessages((prev) => [
    ...prev,
    {
      id: `e-${Date.now()}`,
      role: "assistant",
      content: `Sorry, something went wrong: ${message}. Please try again or check that the Chouchane API is running.`,
    },
  ]);
}

function historyToMessages(history: { role: string; text: string }[]): Message[] {
  return history.map((h, i) => ({
    id: `hist-${i}-${h.role}`,
    role: h.role === "model" ? "assistant" : "user",
    content: h.text,
  }));
}

const FloatingChatbot = () => {
  const { sessionId: ctxSessionId, setSessionId: setCtxSessionId, phase, setPhase: setCtxPhase } = useChouchaneSession();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ctxSessionId) setHistoryLoaded(false);
  }, [ctxSessionId]);

  // When chat opens: use context session if any (and load history), else start new session and save to context
  const ensureSession = useCallback(async () => {
    if (ctxSessionId) {
      if (!historyLoaded && messages.length === 0) {
        setLoading(true);
        try {
          const session = await getSession(ctxSessionId);
          setCtxPhase(session.phase);
          const qaHistory = session.qa_history ?? [];

          // If we're in QA phase and there is no QA history yet,
          // show a friendly welcome message instead of the whole past chat.
          if (session.phase === "qa" && qaHistory.length === 0) {
            setMessages([
              {
                id: "qa-welcome",
                role: "assistant",
                content:
                  "Hello again! Your personalized journey is set. You can now ask me anything related to Tunisia — destinations, culture, food, or tips — I'm here to help.",
              },
            ]);
          } else {
            const fromYasmine =
              session.phase === "yasmine"
                ? historyToMessages(session.yasmine_history ?? [])
                : [];
            const fromQa =
              qaHistory.length > 0
                ? historyToMessages(qaHistory)
                : [];
            setMessages([...fromYasmine, ...fromQa]);
          }
        } catch {
          setMessages([{ id: "1", role: "assistant", content: "Continue the conversation below." }]);
        } finally {
          setLoading(false);
          setHistoryLoaded(true);
        }
      }
      return ctxSessionId;
    }
    setLoading(true);
    try {
      const res = await sessionStart();
      setCtxSessionId(res.session_id);
      setCtxPhase(res.phase);
      setMessages([{ id: "1", role: "assistant", content: res.reply }]);
      return res.session_id;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages([
        {
          id: "1",
          role: "assistant",
          content: `Could not start the travel assistant: ${msg}. Is the Chouchane API running at the configured URL?`,
        },
      ]);
      return null;
    } finally {
      setLoading(false);
    }
  }, [ctxSessionId, historyLoaded, messages.length, setCtxSessionId, setCtxPhase]);

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      ensureSession();
    }
  }, [isOpen, messages.length, ensureSession]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // Listen for "open Chouchen" from Harissa Time (Ask Chouchen for a hint)
  useEffect(() => {
    const handler = (e: CustomEvent<{ spotName?: string; spotId?: number }>) => {
      setIsOpen(true);
      const spotName = e.detail?.spotName;
      const spotId = e.detail?.spotId;
      if (spotName) {
        const hintRequest = `I need a hint for the Harissa challenge at ${spotName}.`;
        setMessages((prev) => [
          ...prev,
          { id: `u-${Date.now()}`, role: "user", content: hintRequest },
        ]);
        ensureSession().then((sid) => {
          if (sid) {
            setLoading(true);
            qaChat(sid, hintRequest)
              .then((r) => {
                setCtxPhase(r.phase);
                appendAssistant(setMessages, r.reply);
                if (spotId != null) {
                  window.dispatchEvent(
                    new CustomEvent("chouchenHint", {
                      detail: { spotId, hint: r.reply },
                    }),
                  );
                }
              })
              .catch((err) =>
                appendError(setMessages, err instanceof Error ? err.message : String(err))
              )
              .finally(() => setLoading(false));
          }
        });
      }
    };
    window.addEventListener("openChouchen", handler as EventListener);
    return () => window.removeEventListener("openChouchen", handler as EventListener);
  }, [ensureSession]);

  const handleSend = async (text?: string) => {
    const msg = text ?? input.trim();
    if (!msg) return;

    const userMsg: Message = { id: `u-${Date.now()}`, role: "user", content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const sid = await ensureSession();
    if (!sid) {
      setLoading(false);
      return;
    }

    const handleResponse = (r: ChouchaneResponse) => {
      setCtxPhase(r.phase);
      appendAssistant(setMessages, r.reply);
      if (r.partners && r.partners.trim()) {
        appendAssistant(setMessages, r.partners.trim());
      }
      setLoading(false);
    };

    try {
      if (phase === "yasmine") {
        const r = await choucheneChat(sid, msg);
        handleResponse(r);
      } else {
        const r = await qaChat(sid, msg);
        handleResponse(r);
      }
    } catch (e) {
      appendError(setMessages, e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  };

  return (
    <>
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-20 right-4 z-[60] w-14 h-14 rounded-full gradient-warm shadow-warm flex items-center justify-center overflow-hidden"
          >
            <MessageCircle size={22} className="text-primary-foreground" />
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-gold flex items-center justify-center">
              <Sparkles size={8} className="text-gold-foreground" />
            </span>
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 100, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 100, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed inset-x-2 bottom-2 top-16 z-[60] max-w-md md:max-w-lg lg:max-w-xl mx-auto flex flex-col rounded-2xl border border-border bg-background shadow-elevated overflow-hidden"
          >
            <div className="relative px-4 py-3 border-b border-border bg-card">
              <TunisianPattern
                variant="stars"
                className="absolute inset-0 w-full h-full text-primary opacity-40"
              />
              <div className="relative flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <img src={chachiaPng} alt="Chouchen" className="w-20 h-20 rounded-full object-contain" />
                  <div>
                    <h3 className="text-sm font-display font-semibold">Chouchen</h3>
                    <p className="text-[10px] text-muted-foreground font-sans">Tunisia travel assistant</p>
                  </div>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center"
                >
                  <X size={14} className="text-muted-foreground" />
                </button>
              </div>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 && loading && (
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Loader2 size={18} className="animate-spin" />
                  Starting your session...
                </div>
              )}
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm font-sans whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "gradient-warm text-primary-foreground rounded-br-md"
                        : "bg-card border border-border text-foreground rounded-bl-md"
                    }`}
                  >
                    {msg.content}
                  </div>
                </motion.div>
              ))}
              {loading && messages.length > 0 && (
                <div className="flex justify-start">
                  <div className="bg-card border border-border rounded-2xl rounded-bl-md px-3.5 py-2.5 flex items-center gap-2 text-muted-foreground text-sm">
                    <Loader2 size={16} className="animate-spin" />
                    Thinking...
                  </div>
                </div>
              )}
            </div>

            {messages.length <= 2 && phase === "yasmine" && (
              <div className="px-4 pb-2 flex gap-2 overflow-x-auto no-scrollbar">
                {quickSuggestions.map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSend(s)}
                    disabled={loading}
                    className="flex-shrink-0 px-3 py-1.5 rounded-full border border-border bg-card text-[11px] font-sans text-muted-foreground hover:text-foreground hover:border-primary/30 transition-colors disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <div className="p-3 border-t border-border bg-card">
              <div className="flex items-center gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                  placeholder="Ask me anything..."
                  disabled={loading}
                  className="flex-1 bg-secondary rounded-xl px-4 py-2.5 text-sm font-sans text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary/30 disabled:opacity-60"
                />
                <button
                  onClick={() => handleSend()}
                  disabled={!input.trim() || loading}
                  className="w-10 h-10 rounded-xl gradient-warm flex items-center justify-center disabled:opacity-40 transition-opacity"
                >
                  <Send size={16} className="text-primary-foreground" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default FloatingChatbot;
