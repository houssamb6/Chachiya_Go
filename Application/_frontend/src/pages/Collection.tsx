import { motion } from "framer-motion";
import { Star, Trophy } from "lucide-react";
import ChachiaCard from "@/components/ChachiaCard";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";

const collected = [
  { name: "Medina Gate", rarity: "standard" as const, xp: 15 },
  { name: "Carthage Gold", rarity: "legendary" as const, xp: 100 },
  { name: "Olive Grove", rarity: "cultural" as const, xp: 50 },
  { name: "Market Square", rarity: "standard" as const, xp: 15 },
  { name: "Sunset Bay", rarity: "rare" as const, xp: 40 },
  { name: "Ancient Well", rarity: "standard" as const, xp: 20 },
];

const achievements = [
  { icon: "🏛", title: "Culture Explorer", desc: "Visit 5 cultural sites", progress: 60 },
  { icon: "🫒", title: "Green Traveler", desc: "Choose 3 sustainable stays", progress: 100 },
  { icon: "🗺", title: "Cartographer", desc: "Unlock 10 map regions", progress: 30 },
];

const Collection = () => {
  const totalXP = 240;
  const level = 4;
  const xpForNext = 300;

  return (
    <div className="app-shell bg-background relative">
      <TunisianPattern variant="stars" className="absolute top-0 left-0 w-full h-64 text-gold opacity-30" />
      <div className="pb-24 relative z-10">
        {/* Header */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <p className="text-xs text-muted-foreground font-sans">Your treasures</p>
          <h1 className="text-xl font-display font-bold">Collection</h1>
        </div>

        {/* XP Card */}
        <div className="px-5 mb-5">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="gradient-warm rounded-2xl p-5 text-primary-foreground"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Star size={18} />
                <span className="text-sm font-semibold font-sans">Level {level}</span>
              </div>
              <span className="text-xs font-sans opacity-80">{totalXP} / {xpForNext} XP</span>
            </div>
            <div className="w-full h-2 rounded-full bg-primary-foreground/20">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(totalXP / xpForNext) * 100}%` }}
                transition={{ delay: 0.3, duration: 0.8 }}
                className="h-full rounded-full bg-primary-foreground/90"
              />
            </div>
            <p className="text-[10px] mt-2 opacity-70 font-sans">
              {xpForNext - totalXP} XP to Level {level + 1}
            </p>
          </motion.div>
        </div>

        {/* Chachias grid */}
        <div className="px-5 mb-6">
          <h3 className="text-sm font-display font-semibold mb-3">Collected Chachias</h3>
          <div className="grid grid-cols-3 gap-3">
            {collected.map((c, i) => (
              <ChachiaCard key={c.name} {...c} delay={i * 0.08} />
            ))}
          </div>
        </div>

        {/* Achievements */}
        <div className="px-5">
          <h3 className="text-sm font-display font-semibold mb-3">Achievements</h3>
          <div className="space-y-3">
            {achievements.map((a, i) => (
              <motion.div
                key={a.title}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 + i * 0.1 }}
                className="flex items-center gap-3 p-3 rounded-xl bg-card border border-border shadow-card"
              >
                <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center text-lg">
                  {a.icon}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold font-sans">{a.title}</p>
                  <p className="text-[10px] text-muted-foreground">{a.desc}</p>
                  <div className="w-full h-1.5 rounded-full bg-border mt-1.5">
                    <div
                      className={`h-full rounded-full transition-all ${
                        a.progress === 100 ? "bg-olive" : "bg-primary"
                      }`}
                      style={{ width: `${a.progress}%` }}
                    />
                  </div>
                </div>
                {a.progress === 100 && (
                  <Trophy size={16} className="text-gold" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </div>
      <BottomNav />
    </div>
  );
};

export default Collection;
