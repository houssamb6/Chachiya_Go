import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Sparkles, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import ChatBubble from "@/components/ChatBubble";
import OptionChip from "@/components/OptionChip";
import chachiaGif from "@/assets/chachia.gif";
import chachiaPng from "@/assets/chachia.png";
import tunisiaFlag from "@/assets/tn.png";
import TunisianPattern from "@/components/TunisianPattern";
import { sessionStart, choucheneChat } from "@/services/chouchaneApi";
import { useChouchaneSession } from "@/context/ChouchaneSessionContext";

interface Message {
  id: number;
  text: string;
  isUser: boolean;
}

const quickSuggestions = [
  { label: "Culture & history", emoji: "🏛" },
  { label: "Beach & relaxation", emoji: "🌊" },
  { label: "Adventure & desert", emoji: "🏜" },
  { label: "Food & local life", emoji: "🍽" },
];

const Onboarding = () => {
  const { sessionId, setSessionId, phase, setPhase } = useChouchaneSession();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const msgId = useRef(0);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const startOnboarding = async () => {
    setShowWelcome(false);
    setLoading(true);
    try {
      const res = await sessionStart();
      setSessionId(res.session_id); // persist in context for rest of app
      setPhase(res.phase);
      msgId.current++;
      setMessages([{ id: msgId.current, text: res.reply, isUser: false }]);
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      msgId.current++;
      setMessages([
        {
          id: msgId.current,
          text: `Could not start: ${errMsg}. Is the Chouchane API running?`,
          isUser: false,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || !sessionId || loading) return;

    msgId.current++;
    setMessages((prev) => [...prev, { id: msgId.current, text: trimmed, isUser: true }]);
    setInput("");
    setLoading(true);

    try {
      const res = await choucheneChat(sessionId, trimmed);

      // Detect transition from Phase 1 (Yasmine) → Phase 2 (QA)
      const wasYasmine = phase === "yasmine";
      const nowQa = res.phase === "qa";

      setPhase(res.phase);

      msgId.current++;
      setMessages((prev) => [...prev, { id: msgId.current, text: res.reply, isUser: false }]);

      if (res.partners && res.partners.trim()) {
        msgId.current++;
        setMessages((prev) => [
          ...prev,
          { id: msgId.current, text: res.partners!.trim(), isUser: false },
        ]);
      }

      // When we *first* enter QA (Phase 1 just ended), keep the final answer visible
      // for 6 seconds, then navigate automatically to the Program page.
      if (wasYasmine && nowQa) {
        msgId.current++;
        setMessages((prev) => [
          ...prev,
          {
            id: msgId.current,
            text: "Your personalized Tunisian journey is ready! You can explore the map, collect Chachias, and ask me anything in the chatbot. Enjoy! ✨",
            isUser: false,
          },
        ]);
        setIsTransitioning(true);
        // Start 6s reading time only after the last message has been painted (not during loading)
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            setTimeout(() => navigate("/home"), 6000);
          });
        });
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      msgId.current++;
      setMessages((prev) => [
        ...prev,
        { id: msgId.current, text: `Something went wrong: ${errMsg}`, isUser: false },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => sendMessage(input);
  const showQuickSuggestions = phase === "yasmine" && messages.length <= 2 && !loading;

  if (showWelcome) {
    return (
      <div className="app-shell bg-[#FAFAF9] relative min-h-screen overflow-hidden">
        <TunisianPattern variant="geometric" className="absolute top-0 left-0 w-64 h-64 text-primary opacity-20" />
        <TunisianPattern variant="arches" className="absolute bottom-24 right-0 w-48 h-48 text-accent opacity-15" />
        <div className="relative h-screen flex flex-col px-6 pt-[max(1.5rem,env(safe-area-inset-top))] pb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="flex items-center gap-2 mb-6"
          >
            <img
              src={tunisiaFlag}
              alt="Tunisia flag"
              className="w-6 h-6 rounded-full"
              loading="lazy"
            />
            <span className="text-xs font-semibold uppercase tracking-widest text-primary font-sans">
              Chacheya Go
            </span>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="flex-1 flex flex-col items-center justify-center max-w-sm mx-auto"
          >
            <div className="rounded-2xl bg-card p-6 mb-6 w-full">
              <div className="flex justify-center mb-4">
                <img src={chachiaGif} alt="Chachia robot waving" className="w-64 h-64 object-contain" />
              </div>
              <h1 className="text-2xl font-display font-bold text-foreground text-center leading-tight">
                Your journey, reimagined.
              </h1>
              <p className="text-sm text-muted-foreground mt-3 font-sans text-center leading-relaxed">
                A personalized, sustainable travel experience crafted by AI — with cultural depth and playful discovery.
              </p>
            </div>

            <div className="flex flex-wrap justify-center gap-3 mb-8">
              <span className="text-xs font-medium text-muted-foreground px-3 py-1.5 rounded-full bg-secondary/80">
                AI-powered
              </span>
              <span className="text-xs font-medium text-muted-foreground px-3 py-1.5 rounded-full bg-secondary/80">
                Sustainable
              </span>
              <span className="text-xs font-medium text-muted-foreground px-3 py-1.5 rounded-full bg-secondary/80">
                Cultural depth
              </span>
            </div>

            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={startOnboarding}
              className="w-full py-4 rounded-2xl gradient-warm text-primary-foreground font-semibold text-base font-sans shadow-warm animate-pulse-glow"
            >
              Begin Your Journey
            </motion.button>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        className="app-shell bg-background"
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
      >
        <div className="flex flex-col h-screen">
          {/* Header */}
          <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-3 border-b border-border bg-card/80 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <img src={chachiaPng} alt="Chachia" className="w-24 h-24 rounded-full object-contain" />
              <div>
                <p className="text-sm font-semibold font-sans">Chouchene</p>
                <p className="text-[10px] text-muted-foreground">
                  {loading ? "Thinking..." : phase === "yasmine" ? "Discovering your preferences..." : "Almost there..."}
                </p>
              </div>
            </div>
            {/* Subtle progress: we're in Chouchene phase until recommendations are done */}
            <div className="flex gap-1.5 mt-3">
              <div
                className={`h-1 flex-1 rounded-full transition-all duration-500 ${
                  phase === "yasmine" ? "bg-primary/60" : "bg-primary"
                }`}
              />
              <div className={`h-1 flex-1 rounded-full transition-all duration-500 ${phase === "qa" ? "bg-primary" : "bg-border"}`} />
            </div>
          </div>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
            {messages.length === 0 && loading && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 size={18} className="animate-spin" />
                Starting your journey with Chouchene...
              </div>
            )}
            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg.text} isUser={msg.isUser} />
            ))}

            {showQuickSuggestions && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-wrap gap-2 pt-2"
              >
                {quickSuggestions.map((opt, i) => (
                  <OptionChip
                    key={opt.label}
                    label={opt.label}
                    emoji={opt.emoji}
                    onClick={() => sendMessage(opt.label)}
                    delay={i * 0.08}
                  />
                ))}
              </motion.div>
            )}
          </div>

          {/* Input */}
          <div className="px-5 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 border-t border-border bg-card/80 backdrop-blur-sm">
            <div className="flex items-center gap-3 bg-secondary rounded-2xl px-4 py-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                placeholder="Tell Chouchene what you like..."
                disabled={loading || phase !== "yasmine" || isTransitioning}
                className="flex-1 bg-transparent text-sm font-sans outline-none text-foreground placeholder:text-muted-foreground disabled:opacity-60"
              />
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!input.trim() || loading || phase !== "yasmine" || isTransitioning}
                className="text-primary disabled:opacity-40 transition-opacity"
              >
                {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default Onboarding;
