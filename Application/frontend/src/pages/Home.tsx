import { motion } from "framer-motion";
import { Sparkles, Bell, Leaf } from "lucide-react";
import ProgramCard from "@/components/ProgramCard";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";
import heroImage from "@/assets/hero-onboarding.jpg";

const sousseActivities = [
  {
    title: "Explore Sousse Medina",
    subtitle: "UNESCO-listed old town, souks & white-and-blue walls",
    image: "https://i0.wp.com/www.engagingcultures.com/wp-content/uploads/2019/09/The-Sousse-Grand-Mosque-Viewed-From-Above.jpg?fit=1000%2C667&ssl=1",
    distance: "City center",
    bestTime: "Morning",
    xp: 45,
    sustainable: true,
  },
  {
    title: "Sousse Beaches",
    subtitle: "Swim, sunbathe & water sports on the Sahel coast",
    image: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&fit=crop",
    distance: "~2 km",
    bestTime: "Afternoon",
    xp: 30,
    sustainable: false,
  },
  {
    title: "Sousse Archaeological Museum",
    subtitle: "Mosaics & Punic-Roman finds from Hadrumetum",
    image: "http://www.commune-sousse.gov.tn/sites/default/files/musee2.gif",
    distance: "~1.5 km",
    bestTime: "Afternoon",
    xp: 55,
    sustainable: true,
  },
  {
    title: "Port El Kantaoui",
    subtitle: "Marina, cafés, golf & boat trips near Sousse",
    image: "https://voyage-tunisie.info/wp-content/uploads/2017/11/Golf-port_El_Kantaoui5.jpg",
    distance: "~10 km",
    bestTime: "Evening",
    xp: 35,
    sustainable: false,
  },
];

const Home = () => {
  return (
    <div className="app-shell bg-background relative">
      <TunisianPattern variant="geometric" className="absolute top-0 right-0 w-48 h-48 text-primary opacity-30" />
      <TunisianPattern variant="arches" className="absolute bottom-32 left-0 w-40 h-40 text-accent opacity-20" />
      <div className="pb-24 relative z-10">
        {/* Header */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-muted-foreground font-sans">Welcome back</p>
              <h1 className="text-xl font-display font-bold">Your Program</h1>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 bg-gold/15 text-gold px-3 py-1.5 rounded-full">
                <Sparkles size={12} />
                <span className="text-xs font-semibold font-sans">240 XP</span>
              </div>
              <button className="w-9 h-9 rounded-full bg-card border border-border flex items-center justify-center">
                <Bell size={16} className="text-foreground" />
              </button>
            </div>
          </div>
        </div>

        {/* Hero card */}
        <div className="px-5 mb-6">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative rounded-2xl overflow-hidden h-40"
          >
            <img src={heroImage} alt="Tunisia" className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-gradient-to-r from-foreground/70 to-transparent" />
            <div className="absolute inset-0 p-5 flex flex-col justify-end relative z-10 -mt-40 h-40">
              <div className="flex items-center gap-1.5 mb-1">
                <Leaf size={12} className="text-olive-foreground" />
                <span className="text-[10px] font-semibold text-olive-foreground uppercase tracking-wider">
                  Sustainable Journey
                </span>
              </div>
              <h2 className="text-lg font-display font-bold text-card">
                Sousse Explorer
              </h2>
              <p className="text-xs text-card/80 font-sans mt-0.5">
                {sousseActivities.length} activities in Sousse • 265 XP potential
              </p>
            </div>
          </motion.div>
        </div>

        {/* Section */}
        <div className="px-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-display font-semibold">Activities in Sousse</h3>
          </div>
          <div className="space-y-3">
            {sousseActivities.map((rec, i) => (
              <ProgramCard key={rec.title} {...rec} delay={i * 0.08} />
            ))}
          </div>
        </div>
      </div>
      <BottomNav />
    </div>
  );
};

export default Home;
