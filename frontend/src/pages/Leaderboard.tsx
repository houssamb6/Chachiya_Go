import { motion } from "framer-motion";
import { Crown, Medal } from "lucide-react";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";
import { useState } from "react";

const leaderboardData = [
  { rank: 1, name: "Amira B.", xp: 2450, level: 12, avatar: "🏛" },
  { rank: 2, name: "Youssef K.", xp: 2100, level: 11, avatar: "🌿" },
  { rank: 3, name: "Leila M.", xp: 1890, level: 10, avatar: "🫒" },
  { rank: 4, name: "You", xp: 240, level: 4, avatar: "🧭", isUser: true },
  { rank: 5, name: "Sami T.", xp: 210, level: 4, avatar: "🏖" },
  { rank: 6, name: "Nour H.", xp: 180, level: 3, avatar: "🍽" },
  { rank: 7, name: "Karim D.", xp: 150, level: 3, avatar: "🏺" },
  { rank: 8, name: "Ines R.", xp: 120, level: 2, avatar: "💎" },
];

const Leaderboard = () => {
  const [tab, setTab] = useState<"weekly" | "global">("weekly");

  return (
    <div className="app-shell bg-background relative">
      <TunisianPattern variant="arches" className="absolute top-0 right-0 w-56 h-56 text-primary opacity-25" />
      <div className="pb-24 relative z-10">
        {/* Header */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <p className="text-xs text-muted-foreground font-sans">Compete</p>
          <h1 className="text-xl font-display font-bold">Leaderboard</h1>
        </div>

        {/* Tabs */}
        <div className="px-5 mb-5">
          <div className="flex bg-secondary rounded-xl p-1">
            {(["weekly", "global"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 py-2 rounded-lg text-xs font-semibold font-sans transition-all ${
                  tab === t
                    ? "bg-card text-foreground shadow-card"
                    : "text-muted-foreground"
                }`}
              >
                {t === "weekly" ? "This Week" : "All Time"}
              </button>
            ))}
          </div>
        </div>

        {/* Top 3 podium */}
        <div className="px-5 mb-6">
          <div className="flex items-end justify-center gap-3">
            {[leaderboardData[1], leaderboardData[0], leaderboardData[2]].map((user, i) => {
              const isFirst = i === 1;
              return (
                <motion.div
                  key={user.rank}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + i * 0.15 }}
                  className={`flex flex-col items-center ${isFirst ? "mb-4" : ""}`}
                >
                  <div className={`relative w-14 h-14 rounded-full flex items-center justify-center text-2xl mb-1 ${
                    isFirst ? "bg-gold/20 border-2 border-gold" : "bg-secondary border-2 border-border"
                  }`}>
                    {user.avatar}
                    {isFirst && (
                      <Crown size={14} className="absolute -top-2 text-gold" />
                    )}
                  </div>
                  <p className="text-xs font-semibold font-sans">{user.name}</p>
                  <p className="text-[10px] text-gold font-semibold">{user.xp} XP</p>
                  <div className={`mt-1 rounded-t-lg flex items-center justify-center text-xs font-bold font-sans ${
                    isFirst
                      ? "w-16 h-20 gradient-warm text-primary-foreground"
                      : user.rank === 2
                      ? "w-14 h-14 bg-secondary text-muted-foreground"
                      : "w-14 h-10 bg-secondary text-muted-foreground"
                  }`}>
                    #{user.rank}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* List */}
        <div className="px-5 space-y-2">
          {leaderboardData.slice(3).map((user, i) => (
            <motion.div
              key={user.rank}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + i * 0.08 }}
              className={`flex items-center gap-3 p-3 rounded-xl border shadow-card ${
                user.isUser
                  ? "bg-primary/5 border-primary/30"
                  : "bg-card border-border"
              }`}
            >
              <span className="w-6 text-xs font-bold text-muted-foreground text-center font-sans">
                #{user.rank}
              </span>
              <div className="w-9 h-9 rounded-full bg-secondary flex items-center justify-center text-lg">
                {user.avatar}
              </div>
              <div className="flex-1">
                <p className={`text-sm font-semibold font-sans ${user.isUser ? "text-primary" : ""}`}>
                  {user.name}
                </p>
                <p className="text-[10px] text-muted-foreground">Level {user.level}</p>
              </div>
              <span className="text-xs font-semibold text-gold font-sans">{user.xp} XP</span>
            </motion.div>
          ))}
        </div>
      </div>
      <BottomNav />
    </div>
  );
};

export default Leaderboard;
