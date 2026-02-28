import { motion } from "framer-motion";
import { MapPin, Clock, Leaf, Star } from "lucide-react";

interface ProgramCardProps {
  title: string;
  subtitle: string;
  image: string;
  distance: string;
  bestTime?: string;
  xp: number;
  sustainable?: boolean;
  delay?: number;
}

const ProgramCard = ({
  title,
  subtitle,
  image,
  distance,
  bestTime,
  xp,
  sustainable,
  delay = 0,
}: ProgramCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="flex gap-3 p-3 rounded-xl bg-card border border-border shadow-card hover:shadow-elevated transition-shadow cursor-pointer"
    >
      <div className="relative w-20 h-20 rounded-lg overflow-hidden flex-shrink-0">
        <img src={image} alt={title} className="w-full h-full object-cover" />
        {sustainable && (
          <div className="absolute top-1 left-1 bg-olive text-olive-foreground rounded-full p-1">
            <Leaf size={10} />
          </div>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-semibold font-display truncate">{title}</h4>
        <p className="text-xs text-muted-foreground mt-0.5 truncate">{subtitle}</p>
        <div className="flex items-center gap-3 mt-2">
          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <MapPin size={10} /> {distance}
          </span>
          {bestTime && (
            <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <Clock size={10} /> {bestTime}
            </span>
          )}
          <span className="flex items-center gap-1 text-[10px] text-gold font-semibold ml-auto">
            <Star size={10} /> +{xp} XP
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default ProgramCard;
