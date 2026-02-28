import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { ChouchaneSessionProvider } from "@/context/ChouchaneSessionContext";
import Index from "./pages/Index";
import Onboarding from "./pages/Onboarding";
import Home from "./pages/Home";
import MapView from "./pages/MapView";
import SpotDetails from "./pages/SpotDetails";
import Collection from "./pages/Collection";
import Leaderboard from "./pages/Leaderboard";
import Rewards from "./pages/Rewards";
import NotFound from "./pages/NotFound";
import FloatingChatbot from "./components/FloatingChatbot";

const queryClient = new QueryClient();

const AppContent = () => {
  const location = useLocation();
  const showChatbot = !["/", "/onboarding"].includes(location.pathname);

  return (
    <>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/home" element={<Home />} />
        <Route path="/map" element={<MapView />} />
        <Route path="/spot/:id" element={<SpotDetails />} />
        <Route path="/collection" element={<Collection />} />
        <Route path="/leaderboard" element={<Leaderboard />} />
        <Route path="/rewards" element={<Rewards />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      {showChatbot && <FloatingChatbot />}
    </>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ChouchaneSessionProvider>
          <AppContent />
        </ChouchaneSessionProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
