import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, MapPin, Calendar, ImageIcon } from "lucide-react";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";
import { getChachiaSpotById } from "@/data/chachiaSpots";
import { Button } from "@/components/ui/button";

const SpotDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const spot = id ? getChachiaSpotById(parseInt(id, 10)) : undefined;

  if (!spot) {
    return (
      <div className="app-shell bg-background relative min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground mb-4">Spot not found</p>
          <Button variant="outline" onClick={() => navigate("/map")}>
            Back to Map
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell bg-background relative">
      <TunisianPattern variant="stars" className="absolute top-0 right-0 w-40 h-40 text-accent opacity-20" />
      <div className="pb-24 relative z-10">
        {/* Header with back button */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <button
            onClick={() => navigate("/map")}
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-4"
          >
            <ArrowLeft size={20} />
            <span className="text-sm font-medium">Back to Map</span>
          </button>
          <div className="flex items-center gap-3">
            <img
              src={spot.image}
              alt={spot.name}
              className="w-14 h-14 object-contain rounded-full border-2 border-border shadow-md"
            />
            <div>
              <h1 className="text-xl font-display font-bold">{spot.name}</h1>
              <span
                className={`inline-flex text-xs font-semibold px-2 py-0.5 rounded-full ${
                  spot.rarity === "Legendary"
                    ? "bg-amber-100 text-amber-800"
                    : spot.rarity === "Cultural"
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-slate-100 text-slate-700"
                }`}
              >
                {spot.rarity}
              </span>
            </div>
          </div>
        </div>

        <div className="px-5 space-y-6">
          {/* Images gallery */}
          {spot.imageUrls.length > 0 && (
            <motion.section
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="space-y-3"
            >
              <h2 className="flex items-center gap-2 text-sm font-display font-semibold">
                <ImageIcon size={18} />
                Images
              </h2>
              <div className="space-y-3">
                {spot.imageUrls.map((url, idx) => (
                  <motion.div
                    key={url}
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.15 + idx * 0.05 }}
                    className="rounded-xl overflow-hidden border border-border shadow-card bg-card"
                  >
                    <img
                      src={url}
                      alt={`${spot.name} view ${idx + 1}`}
                      className="w-full h-48 object-cover"
                      loading="lazy"
                    />
                  </motion.div>
                ))}
              </div>
            </motion.section>
          )}

          {/* Details */}
          <motion.section
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-3"
          >
            <h2 className="flex items-center gap-2 text-sm font-display font-semibold">
              <MapPin size={18} />
              About this spot
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">{spot.description}</p>
          </motion.section>

          {/* History */}
          <motion.section
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-3"
          >
            <h2 className="flex items-center gap-2 text-sm font-display font-semibold">
              <Calendar size={18} />
              History
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">{spot.history}</p>
          </motion.section>

          {/* Location hint */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Button
              variant="outline"
              className="w-full"
              onClick={() => navigate("/map")}
            >
              <MapPin size={16} />
              View on Map
            </Button>
          </motion.div>
        </div>
      </div>
      <BottomNav />
    </div>
  );
};

export default SpotDetails;
