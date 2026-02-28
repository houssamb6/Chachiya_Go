import { motion } from "framer-motion";
import { Gift, Lock, Sparkles, Star, ChevronRight } from "lucide-react";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";

const rewards = [
  {
    title: "Spa Day at Dar El Marsa",
    desc: "Full day hammam & massage experience",
    xpRequired: 200,
    category: "Wellness",
    emoji: "🧖",
    unlocked: true,
  },
  {
    title: "15% off at Dar El Jeld",
    desc: "Fine dining discount at iconic restaurant",
    xpRequired: 150,
    category: "Dining",
    emoji: "🍽",
    unlocked: true,
  },
  {
    title: "Guided Medina Tour",
    desc: "Private 2-hour walking tour with local expert",
    xpRequired: 300,
    category: "Experience",
    emoji: "🏛",
    unlocked: false,
  },
  {
    title: "Pottery Workshop",
    desc: "Hands-on ceramic class in Nabeul",
    xpRequired: 400,
    category: "Culture",
    emoji: "🏺",
    unlocked: false,
  },
  {
    title: "Sunset Sailing Trip",
    desc: "Private catamaran cruise along Cap Bon",
    xpRequired: 500,
    category: "Adventure",
    emoji: "⛵",
    unlocked: false,
  },
  {
    title: "VIP Desert Experience",
    desc: "Overnight luxury glamping in Tozeur",
    xpRequired: 800,
    category: "Premium",
    emoji: "🏜",
    unlocked: false,
  },
];

const Rewards = () => {
  const userXP = 240;
  const unlockedRewards = rewards.filter((r) => r.unlocked);
  const lockedRewards = rewards.filter((r) => !r.unlocked);
  const nextReward = lockedRewards[0];
  const progressToNext = nextReward
    ? Math.min(100, (userXP / nextReward.xpRequired) * 100)
    : 100;

  return (
    <div className="app-shell bg-background relative min-h-screen">
      <TunisianPattern
        variant="geometric"
        className="absolute bottom-40 right-0 w-52 h-52 text-gold opacity-20"
      />
      <TunisianPattern
        variant="stars"
        className="absolute top-20 left-0 w-40 h-40 text-primary opacity-15"
      />
      <div className="pb-24 relative z-10">
        {/* Header */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <p className="text-xs text-muted-foreground font-sans uppercase tracking-wider">
            Redeem your journey
          </p>
          <h1 className="text-2xl font-display font-bold text-foreground">Rewards</h1>
        </div>

        {/* XP hero card */}
        <div className="px-5 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="relative rounded-2xl overflow-hidden"
          >
            <div className="absolute inset-0 gradient-warm opacity-95" />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(255,255,255,0.25),transparent)]" />
            <div className="relative p-5 text-primary-foreground">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <Gift size={24} />
                  </div>
                  <div>
                    <p className="text-sm font-medium opacity-90">Your balance</p>
                    <p className="text-2xl font-display font-bold tracking-tight">{userXP} XP</p>
                  </div>
                </div>
                <Star size={20} className="opacity-80" />
              </div>
              {nextReward && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="opacity-90">Next: {nextReward.title}</span>
                    <span className="font-semibold">{nextReward.xpRequired} XP</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/25 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${progressToNext}%` }}
                      transition={{ delay: 0.3, duration: 0.8, ease: "easeOut" }}
                      className="h-full rounded-full bg-white/90"
                    />
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Unlocked rewards */}
        {unlockedRewards.length > 0 && (
          <div className="px-5 mb-6">
            <h2 className="text-sm font-display font-semibold text-foreground mb-3 flex items-center gap-2">
              <Sparkles size={16} className="text-gold" />
              Ready to claim
            </h2>
            <div className="space-y-3">
              {unlockedRewards.map((r, i) => (
                <motion.div
                  key={r.title}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 + i * 0.06 }}
                  className="group flex items-center gap-4 p-4 rounded-2xl bg-card border border-border shadow-card hover:shadow-elevated hover:border-primary/20 transition-all"
                >
                  <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center text-2xl flex-shrink-0">
                    {r.emoji}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-display font-semibold text-foreground truncate">
                      {r.title}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{r.desc}</p>
                    <span className="inline-block mt-2 text-[10px] font-medium text-primary bg-primary/10 px-2 py-1 rounded-full">
                      {r.category} • {r.xpRequired} XP
                    </span>
                  </div>
                  <button className="flex items-center gap-1 px-4 py-2.5 rounded-xl gradient-warm text-primary-foreground text-xs font-semibold shadow-warm hover:opacity-95 transition-opacity">
                    Claim
                    <ChevronRight size={14} />
                  </button>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Locked rewards */}
        <div className="px-5">
          <h2 className="text-sm font-display font-semibold text-foreground mb-3 flex items-center gap-2">
            <Lock size={14} className="text-muted-foreground" />
            Keep exploring
          </h2>
          <div className="space-y-3">
            {lockedRewards.map((r, i) => {
              const progress = Math.min(100, (userXP / r.xpRequired) * 100);
              return (
                <motion.div
                  key={r.title}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + i * 0.06 }}
                  className="flex items-center gap-4 p-4 rounded-2xl bg-muted/40 border border-border opacity-90"
                >
                  <div className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center text-2xl flex-shrink-0">
                    {r.emoji}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <h3 className="text-sm font-display font-semibold text-foreground truncate">
                        {r.title}
                      </h3>
                      <Lock size={12} className="text-muted-foreground flex-shrink-0" />
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{r.desc}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full bg-secondary overflow-hidden max-w-[100px]">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${progress}%` }}
                          transition={{ delay: 0.4 + i * 0.05, duration: 0.6 }}
                          className="h-full rounded-full bg-gold/60"
                        />
                      </div>
                      <span className="text-[10px] font-semibold text-muted-foreground">
                        {userXP}/{r.xpRequired} XP
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
      <BottomNav />
    </div>
  );
};

export default Rewards;
