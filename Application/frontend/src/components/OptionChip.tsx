import { motion } from "framer-motion";

interface OptionChipProps {
  label: string;
  emoji?: string;
  selected?: boolean;
  onClick: () => void;
  delay?: number;
}

const OptionChip = ({ label, emoji, selected, onClick, delay = 0 }: OptionChipProps) => {
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.25 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium font-sans border transition-all duration-200 ${
        selected
          ? "bg-primary text-primary-foreground border-primary shadow-warm"
          : "bg-card text-foreground border-border hover:border-primary/40 shadow-card"
      }`}
    >
      {emoji && <span>{emoji}</span>}
      {label}
    </motion.button>
  );
};

export default OptionChip;
