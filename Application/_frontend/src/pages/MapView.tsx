import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { MapPin, Navigation, Loader2, AlertCircle, Sparkles, Trophy, Flame, Lightbulb } from "lucide-react";
import L from "leaflet";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import type { Map as LeafletMap } from "leaflet";
import BottomNav from "@/components/BottomNav";
import TunisianPattern from "@/components/TunisianPattern";
import "leaflet/dist/leaflet.css";
import { chachiaSpots } from "@/data/chachiaSpots";

// Tunisia center coordinates
const TUNISIA_CENTER = { lat: 36.8065, lng: 10.1615 };
const COLLECTION_RADIUS_KM = 0.5; // 500 meters

// Haversine formula to calculate distance between two coordinates
const calculateDistance = (
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number => {
  const R = 6371; // Earth's radius in kilometers
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

const MapView = () => {
  const navigate = useNavigate();
  const [userPosition, setUserPosition] = useState<{ lat: number; lng: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [collectedChachias, setCollectedChachias] = useState<Set<number>>(new Set());
  const [totalXP, setTotalXP] = useState(0);
  const [celebrationData, setCelebrationData] = useState<{
    show: boolean;
    name: string;
    xp: number;
    image: string;
    rarity: string;
  } | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [showHarissaModal, setShowHarissaModal] = useState(false);
  const [activeHarissaSpot, setActiveHarissaSpot] = useState<(typeof chachiaSpots)[0] | null>(null);
  const [harissaGuess, setHarissaGuess] = useState("");
  const [harissaFeedback, setHarissaFeedback] = useState<"correct" | "wrong" | null>(null);
  const [harissaCompleted, setHarissaCompleted] = useState<Set<number>>(new Set());
  const [harissaHint, setHarissaHint] = useState<string | null>(null);

  // Custom icon for user position - clean map pin
  const userIcon = L.divIcon({
    html: `
      <div style="display: flex; flex-direction: column; align-items: center;">
        <svg width="32" height="40" viewBox="0 0 24 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 9 12 24 12 24s12-15 12-24C24 5.37 18.63 0 12 0z" fill="#2563eb"/>
          <circle cx="12" cy="12" r="5" fill="white"/>
        </svg>
      </div>
    `,
    iconSize: [32, 40],
    iconAnchor: [16, 40],
    popupAnchor: [0, -40],
    className: "custom-user-icon",
  });

  // Custom icons for chachia markers
  const createChachiaIcon = (image: string, rarity: string) => {
    const borderColors: { [key: string]: string } = {
      Standard: "hsl(0, 0%, 88%)",
      Rare: "hsl(45, 93%, 50%)",
      Legendary: "hsl(45, 93%, 50%)",
      Cultural: "hsl(142, 71%, 45%)",
    };
    const bgColors: { [key: string]: string } = {
      Standard: "#f5f5f5",
      Rare: "#fef3c7",
      Legendary: "#fef3c7",
      Cultural: "#dcfce7",
    };
    const borderColor = borderColors[rarity] || "#d1d5db";
    const bgColor = bgColors[rarity] || "#f9fafb";

    return L.divIcon({
      html: `
        <div class="flex flex-col items-center">
          <div class="relative bg-white rounded-full shadow-lg overflow-hidden" style="width: 64px; height: 64px; border: 3px solid ${borderColor}; background: ${bgColor}; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
            <img src="${image}" alt="chachia" style="width: 100%; height: 100%; object-fit: contain; padding: 2px;" class="animate-fade-in" />
          </div>
          <div class="mt-0.5 bg-gradient-to-r from-amber-500 to-orange-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-md whitespace-nowrap">
            ${rarity}
          </div>
        </div>
      `,
      iconSize: [44, 68],
      iconAnchor: [22, 68],
      popupAnchor: [0, -68],
      className: "custom-chachia-icon",
    });
  };

  const openHarissaForSpot = (spot: (typeof chachiaSpots)[0]) => {
    if (!spot.harissa) return;
    setActiveHarissaSpot(spot);
    setHarissaGuess("");
    setHarissaFeedback(null);
    setHarissaHint(null);
    setShowHarissaModal(true);
  };

  const handleSubmitHarissaGuess = () => {
    if (!activeHarissaSpot?.harissa) return;
    const guess = harissaGuess.trim().toLowerCase();
    if (!guess) return;
    const answer = activeHarissaSpot.harissa.answer.trim().toLowerCase();
    if (guess === answer) {
      setHarissaFeedback("correct");
      setHarissaCompleted((prev) => new Set([...prev, activeHarissaSpot.id]));
    } else {
      setHarissaFeedback("wrong");
    }
  };

  const handleAskChouchenForHint = () => {
    if (!activeHarissaSpot?.harissa) return;
    window.dispatchEvent(
      new CustomEvent("openChouchen", {
        detail: { spotName: activeHarissaSpot.name, spotId: activeHarissaSpot.id },
      }),
    );
  };

  // Receive hint text back from Chouchen and surface it inside the Harissa card
  useEffect(() => {
    const handler = (e: CustomEvent<{ spotId?: number; hint: string }>) => {
      const spotId = e.detail?.spotId;
      if (!activeHarissaSpot || (spotId != null && spotId !== activeHarissaSpot.id)) return;
      setHarissaHint(e.detail.hint);
    };
    window.addEventListener("chouchenHint", handler as EventListener);
    return () => window.removeEventListener("chouchenHint", handler as EventListener);
  }, [activeHarissaSpot]);

  const handleCollectChachia = (marker: (typeof chachiaSpots)[0]) => {
    if (!collectedChachias.has(marker.id) && userPosition) {
      setCollectedChachias((prev) => new Set([...prev, marker.id]));
      setTotalXP((prev) => prev + marker.xp);
      setCelebrationData({
        show: true,
        name: marker.name,
        xp: marker.xp,
        image: marker.image,
        rarity: marker.rarity,
      });
      // Auto-hide celebration after 3.5 seconds
      setTimeout(() => setCelebrationData(null), 3500);
      // Trigger Harissa Time popup shortly after celebrating, for spots that have a challenge
      if (marker.harissa && !harissaCompleted.has(marker.id)) {
        setTimeout(() => {
          openHarissaForSpot(marker);
        }, 3800);
      }
    }
  };

  const handleCenterMap = () => {
    if (mapRef.current && userPosition) {
      setIsLocating(true);
      mapRef.current.setView([userPosition.lat, userPosition.lng], 14, {
        animate: true,
        duration: 1,
      });
      setTimeout(() => setIsLocating(false), 1000);
    }
  };

  const handleFocusSpot = (lat: number, lng: number) => {
    if (mapRef.current) {
      mapRef.current.setView([lat, lng], 15, {
        animate: true,
        duration: 0.8,
      });
    }
  };

  useEffect(() => {
    // Request user's location
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserPosition({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
          setLoading(false);
        },
        (err) => {
          console.log("Location access denied, using Tunisia center");
          // Use Tunisia center if permission denied
          setUserPosition(TUNISIA_CENTER);
          setError("Location access denied. Showing Tunisia map.");
          setLoading(false);
        }
      );
    } else {
      setUserPosition(TUNISIA_CENTER);
      setError("Geolocation not supported");
      setLoading(false);
    }
  }, []);

  if (loading) {
    return (
      <div className="app-shell bg-background relative">
        <div className="pb-24 relative z-10 flex flex-col items-center justify-center min-h-screen">
          <Loader2 size={32} className="animate-spin text-primary mb-4" />
          <p className="text-muted-foreground">Getting your location...</p>
        </div>
        <BottomNav />
      </div>
    );
  }

  return (
    <div className="app-shell bg-background relative">
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 8px rgba(59, 130, 246, 0.6); }
          50% { box-shadow: 0 0 14px rgba(59, 130, 246, 0.9); }
        }
        .custom-user-icon {
          filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        }
        .custom-user-icon > div > div > div {
          animation: pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        .custom-chachia-icon {
          filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.2));
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in {
          animation: fade-in 0.3s ease-in;
        }
        @keyframes confetti-fall {
          0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 1;
          }
          100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
          }
        }
        .confetti {
          animation: confetti-fall 3s ease-in forwards;
          position: fixed;
          pointer-events: none;
        }
        @keyframes bounce-scale {
          0%, 100% { transform: scale(0); }
          50% { transform: scale(1.2); }
        }
        .celebration-icon {
          animation: bounce-scale 0.6s cubic-bezier(0.36, 0, 0.66, -0.56);
        }
        .leaflet-popup-content {
          font-family: inherit;
        }
      `}</style>
      <TunisianPattern variant="stars" className="absolute top-0 right-0 w-40 h-40 text-accent opacity-20" />
      <div className="pb-24 relative z-10">
        {/* Header */}
        <div className="px-5 pt-[max(1rem,env(safe-area-inset-top))] pb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-muted-foreground font-sans">Explore</p>
              <h1 className="text-xl font-display font-bold">Discovery Map</h1>
            </div>
            <button
              onClick={handleCenterMap}
              disabled={isLocating}
              className="w-9 h-9 rounded-full bg-card border border-border flex items-center justify-center hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Center map on your location"
            >
              <Navigation size={16} className={`text-primary transition-transform ${isLocating ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* Error message if any */}
        {/* {error && (
          <div className="px-5 mb-3">
            <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/30 rounded-lg p-2.5">
              <AlertCircle size={16} className="text-amber-600 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-amber-700">{error}</p>
            </div>
          </div>
        )} */}

        {/* Celebration Modal */}
        <AnimatePresence>
          {celebrationData?.show && (
            <motion.div
              initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/60 backdrop-blur-md z-[99999] flex items-center justify-center pointer-events-none"
            >
              {/* Animated background flash */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 0.3, 0] }}
                transition={{ duration: 0.6 }}
                className="absolute inset-0 bg-white"
              />

              <motion.div
                initial={{ scale: 0, y: -200 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0, y: 200 }}
                transition={{ type: "spring", stiffness: 200, damping: 25 }}
                className="bg-gradient-to-b from-primary via-primary/90 to-primary/80 rounded-3xl p-8 text-center text-white pointer-events-auto relative overflow-hidden max-w-md md:max-w-lg lg:max-w-xl mx-4"
              >
                {/* Animated background glow */}
                <motion.div
                  animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute inset-0 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-3xl blur-xl"
                  style={{ zIndex: -1 }}
                />

                {/* Confetti effects */}
                {[...Array(16)].map((_, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 1, y: -20, x: 0 }}
                    animate={{
                      opacity: 0,
                      y: 300,
                      x: Math.sin(i) * 100,
                      rotate: 360,
                    }}
                    transition={{ duration: 2.5, delay: i * 0.05 }}
                    className="confetti"
                    style={{
                      left: `${Math.random() * 100}%`,
                      backgroundColor: ["#FFD700", "#FF6B6B", "#4ECDC4", "#95E1D3", "#FFB700"][i % 5],
                      width: "14px",
                      height: "14px",
                      borderRadius: "50%",
                    }}
                  />
                ))}

                {/* Main content */}
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 300 }}
                  className="mb-4"
                >
                  <Trophy className="w-16 h-16 mx-auto text-yellow-300 drop-shadow-lg" />
                </motion.div>

                <motion.h1
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.25 }}
                  className="text-5xl font-black mb-1 tracking-tight"
                >
                  CONGRATS!
                </motion.h1>

                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.35, duration: 0.5 }}
                  className="h-1 bg-gradient-to-r from-yellow-300 via-orange-300 to-yellow-300 rounded-full mb-4 mx-auto w-24"
                />

                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="text-xl font-semibold mb-4"
                >
                  You found a new Chachia!
                </motion.p>

                {/* Chachia info card */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.8, rotateX: 90 }}
                  animate={{ opacity: 1, scale: 1, rotateX: 0 }}
                  transition={{ delay: 0.5, type: "spring" }}
                  className="flex items-center justify-center gap-3 mb-5 bg-white/25 backdrop-blur rounded-2xl p-4 border border-white/30"
                >
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    <img src={celebrationData.image} alt={celebrationData.name} className="w-12 h-12 object-contain" />
                  </motion.div>
                  <div className="text-left">
                    <div className="font-bold text-lg">{celebrationData.name}</div>
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.6 }}
                      className={`text-sm font-semibold ${
                        celebrationData.rarity === "Legendary" ? "text-yellow-300" :
                        celebrationData.rarity === "Cultural" ? "text-emerald-300" :
                        "text-blue-200"
                      }`}
                    >
                      {celebrationData.rarity}
                    </motion.div>
                  </div>
                </motion.div>

                {/* XP display with animation */}
                <motion.div
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.65, type: "spring", stiffness: 200 }}
                  className="flex items-center justify-center gap-2 text-3xl font-black mb-3 bg-white/15 rounded-2xl p-3"
                >
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Sparkles className="w-8 h-8 text-yellow-300" />
                  </motion.div>
                  <span>+{celebrationData.xp} XP</span>
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Sparkles className="w-8 h-8 text-yellow-300" />
                  </motion.div>
                </motion.div>

                {/* Total XP */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7 }}
                  className="text-base text-white/90 pt-3 border-t border-white/20"
                >
                  Total XP: <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.75, type: "spring" }}
                    className="font-black text-2xl text-yellow-300"
                  >
                    {totalXP}
                  </motion.span>
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Harissa Time modal */}
        <AnimatePresence>
          {showHarissaModal && activeHarissaSpot?.harissa && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-md z-[9999] flex items-center justify-center"
            >
              <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.9, y: 20 }}
                transition={{ type: "spring", stiffness: 220, damping: 22 }}
                className="bg-card rounded-3xl p-5 mx-4 max-w-md md:max-w-lg lg:max-w-xl w-full border border-border shadow-elevated relative"
              >
                <button
                  onClick={() => setShowHarissaModal(false)}
                  className="absolute top-3 right-3 text-xs text-muted-foreground hover:text-foreground"
                >
                  Close
                </button>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-full bg-red-500/10 flex items-center justify-center text-red-600">
                    <Flame size={18} />
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-widest text-red-600">
                      Harissa time
                    </p>
                    <p className="text-sm font-display font-semibold">
                      {activeHarissaSpot.name}
                    </p>
                  </div>
                </div>

                <p className="text-sm text-foreground mb-3 leading-relaxed">
                  {activeHarissaSpot.harissa.question}
                </p>

                <div className="space-y-2 mb-3">
                  <input
                    type="text"
                    value={harissaGuess}
                    onChange={(e) => setHarissaGuess(e.target.value)}
                    placeholder="Type your mysterious word..."
                    className="w-full rounded-xl border border-border bg-secondary px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-primary/40"
                  />
                  <button
                    onClick={handleSubmitHarissaGuess}
                    className="w-full py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors"
                  >
                    Check my guess
                  </button>
                </div>

                {harissaFeedback === "correct" && (
                  <div className="mt-2 rounded-xl bg-emerald-500/10 border border-emerald-500/40 px-3 py-2 flex items-start gap-2 text-xs text-emerald-700">
                    <Lightbulb size={14} className="mt-0.5" />
                    <div>
                      <p className="font-semibold">Bravo! 🔥</p>
                      <p>You solved this Harissa Time. You can revisit it anytime from the chachia popup.</p>
                    </div>
                  </div>
                )}

                {harissaFeedback === "wrong" && (
                  <div className="mt-2 rounded-xl bg-amber-500/10 border border-amber-500/40 px-3 py-2 text-xs text-amber-800">
                    <p className="font-semibold mb-1">
                      Not quite... Chouchen can help with up to 3 hints.
                    </p>
                    <button
                      onClick={handleAskChouchenForHint}
                      className="mt-1 inline-flex items-center gap-1 rounded-full border border-amber-500/60 px-3 py-1 text-[11px] font-medium bg-amber-50/60 hover:bg-amber-100/60 transition-colors"
                    >
                      <Sparkles size={12} />
                      Ask Chouchen for a hint
                    </button>
                    {harissaHint && (
                      <p className="mt-2 text-[11px] text-amber-900 leading-relaxed">
                        Hint from Chouchen: {harissaHint}
                      </p>
                    )}
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Map area */}
        <div className="px-5">
          <div className="relative rounded-2xl overflow-hidden bg-secondary border border-border h-[55vh]">
            {userPosition && (
              <MapContainer
                ref={mapRef}
                center={[userPosition.lat, userPosition.lng]}
                zoom={10}
                style={{ height: "100%", width: "100%" }}
                className="rounded-2xl"
              >
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />

                {/* User position marker - zIndexOffset so it stays on top of chachia markers */}
                <Marker position={[userPosition.lat, userPosition.lng]} icon={userIcon} zIndexOffset={1000}>
                  <Popup>
                    <div className="text-sm font-medium">Your Location</div>
                    <div className="text-xs text-muted-foreground">
                      ({userPosition.lat.toFixed(4)}, {userPosition.lng.toFixed(4)})
                    </div>
                  </Popup>
                </Marker>

                {/* Chachia markers */}
                {chachiaSpots.map((marker) => {
                  const distance = userPosition ? calculateDistance(userPosition.lat, userPosition.lng, marker.lat, marker.lng) : null;
                  const canCollect = distance !== null && distance <= COLLECTION_RADIUS_KM && !collectedChachias.has(marker.id);
                  const isCollected = collectedChachias.has(marker.id);

                  return (
                    <Marker
                      key={marker.id}
                      position={[marker.lat, marker.lng]}
                      icon={createChachiaIcon(marker.image, marker.rarity)}
                    >
                      <Popup>
                        <div className="w-48">
                          <div className="flex gap-2 mb-2">
                            <img src={marker.image} alt={marker.name} className="w-12 h-12 object-contain" />
                            <div>
                              <div className="font-medium text-sm">{marker.name}</div>
                              <div className={`text-xs font-semibold ${
                                marker.rarity === "Legendary" ? "text-amber-600" :
                                marker.rarity === "Cultural" ? "text-emerald-600" :
                                "text-slate-600"
                              }`}>
                                {marker.rarity}
                              </div>
                            </div>
                          </div>
                          <div className="text-xs text-muted-foreground mb-2">
                            📍 ({marker.lat.toFixed(4)}, {marker.lng.toFixed(4)})
                          </div>
                          {distance !== null && (
                            <div className={`text-xs font-semibold mb-2 ${canCollect ? "text-green-600" : "text-slate-600"}`}>
                              📏 {distance.toFixed(2)} km away
                            </div>
                          )}
                          {isCollected && (
                            <div className="w-full bg-green-500 text-white text-xs py-1.5 rounded font-medium flex items-center justify-center gap-1">
                              ✓ Collected
                            </div>
                          )}
                          {canCollect && (
                            <button
                              onClick={() => handleCollectChachia(marker)}
                              className="w-full bg-primary text-primary-foreground text-xs py-1.5 rounded hover:bg-primary/90 transition-colors font-medium"
                            >
                              Collect +{marker.xp} XP
                            </button>
                          )}
                          {marker.harissa && isCollected && (
                            <button
                              onClick={() => openHarissaForSpot(marker)}
                              className="w-full mt-2 bg-amber-500 text-white text-xs py-1.5 rounded hover:bg-amber-600 transition-colors font-medium"
                            >
                              Harissa time
                            </button>
                          )}
                          <button
                            onClick={() => navigate(`/spot/${marker.id}`)}
                            className="w-full mt-2 bg-secondary border border-border text-foreground text-xs py-1.5 rounded hover:bg-accent transition-colors font-medium"
                          >
                            More details
                          </button>
                          {!canCollect && !isCollected && (
                            <div className="w-full bg-slate-300 text-slate-700 text-xs py-1.5 rounded text-center font-medium cursor-not-allowed">
                              Get closer ({(COLLECTION_RADIUS_KM - (distance || 0)).toFixed(2)} km)
                            </div>
                          )}
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}
              </MapContainer>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="px-5 mt-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-display font-semibold">Nearby Chachias</h3>
            <div className="text-sm font-bold text-primary">Total XP: {totalXP}</div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[...chachiaSpots]
              .sort((a, b) => {
                if (!userPosition) return 0;
                const distA = calculateDistance(userPosition.lat, userPosition.lng, a.lat, a.lng);
                const distB = calculateDistance(userPosition.lat, userPosition.lng, b.lat, b.lng);
                return distA - distB;
              })
              .map((marker) => {
              const distance = userPosition ? calculateDistance(userPosition.lat, userPosition.lng, marker.lat, marker.lng) : null;
              const canCollect = distance !== null && distance <= COLLECTION_RADIUS_KM && !collectedChachias.has(marker.id);
              const isCollected = collectedChachias.has(marker.id);

              return (
                <motion.div
                  key={marker.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleFocusSpot(marker.lat, marker.lng)}
                  onKeyDown={(e) => e.key === "Enter" && handleFocusSpot(marker.lat, marker.lng)}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex items-start gap-2 p-2.5 rounded-lg border transition-all cursor-pointer hover:opacity-90 ${
                    isCollected
                      ? "bg-green-500/10 border-green-500/30 opacity-60"
                      : canCollect
                      ? "bg-primary/10 border-primary/30 ring-1 ring-primary"
                      : marker.rarity === "Legendary"
                      ? "bg-gold/10 border-gold/30"
                      : marker.rarity === "Cultural"
                      ? "bg-emerald-500/10 border-emerald-500/30"
                      : "bg-card border-border"
                  }`}
                >
                  <div className="relative flex-shrink-0">
                    <img src={marker.image} alt={marker.name} className="w-6 h-6 object-contain" />
                    {isCollected && (
                      <div className="absolute -top-1 -right-1 bg-green-500 text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px]">
                        ✓
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{marker.name}</div>
                    <div className={`text-[10px] ${isCollected ? "text-green-600 font-semibold" : "text-muted-foreground"}`}>
                      {isCollected
                        ? "✓ Collected"
                        : distance !== null
                        ? `${distance.toFixed(2)} km ${canCollect ? "✨ Ready!" : ""}`
                        : "+{marker.xp} XP"}
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

export default MapView;
