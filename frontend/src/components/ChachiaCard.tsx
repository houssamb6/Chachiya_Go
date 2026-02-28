import { motion } from "framer-motion";
import chachiaStandard from "@/assets/chachia-standard.png";
import chachiaRare from "@/assets/chachia-rare.png";
import chachiaLegendary from "@/assets/chachia-legendary.png";
import chachiaCultural from "@/assets/chachia-cultural.png";

type Rarity = "standard" | "rare" | "legendary" | "cultural";

const rarityConfig: Record<Rarity, { label: string; color: string; bg: string; image: string }> = {
  standard: { label: "Standard", color: "text-muted-foreground", bg: "bg-secondary", image: chachiaStandard },
  rare: { label: "Rare", color: "text-primary", bg: "bg-primary/10", image: chachiaRare },
  legendary: { label: "Legendary", color: "text-gold", bg: "bg-gold/10", image: chachiaLegendary },
  cultural: { label: "Cultural", color: "text-olive", bg: "bg-olive/10", image: chachiaCultural },
};

interface ChachiaCardProps {
  name: string;
  rarity: Rarity;
  xp: number;
  collected?: boolean;
  delay?: number;
}

const ChachiaCard = ({ name, rarity, xp, collected = true, delay = 0 }: ChachiaCardProps) => {
  const config = rarityConfig[rarity];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.3 }}
      className={`relative rounded-xl border border-border p-3 flex flex-col items-center gap-2 ${
        collected ? "bg-card shadow-card" : "bg-muted opacity-50"
      }`}
    >
      {rarity === "legendary" && collected && (
        <div className="absolute inset-0 rounded-xl animate-shimmer pointer-events-none" />
      )}
      <div className={`w-14 h-14 rounded-full ${config.bg} flex items-center justify-center overflow-hidden`}>
        <img src={config.image} alt={name} className="w-10 h-10 object-contain" />
      </div>
      <p className="text-xs font-semibold font-sans text-center truncate w-full">{name}</p>
      <div className="flex items-center gap-1">
        <span className={`text-[10px] font-semibold ${config.color}`}>{config.label}</span>
        <span className="text-[10px] text-gold font-semibold">+{xp} XP</span>
      </div>
    </motion.div>
  );
};

export default ChachiaCard;
