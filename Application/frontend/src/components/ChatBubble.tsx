import { motion } from "framer-motion";

interface ChatBubbleProps {
  message: string;
  isUser?: boolean;
  delay?: number;
}

const ChatBubble = ({ message, isUser = false, delay = 0 }: ChatBubbleProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay, duration: 0.35, ease: "easeOut" }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed font-sans whitespace-pre-wrap ${
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md"
            : "bg-card text-card-foreground rounded-bl-md shadow-card border border-border"
        }`}
      >
        {message}
      </div>
    </motion.div>
  );
};

export default ChatBubble;
