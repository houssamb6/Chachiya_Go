import chachiaStandard from "@/assets/chachia-standard.png";
import chachiaLegendary from "@/assets/chachia-legendary.png";
import chachiaCultural from "@/assets/chachia-cultural.png";

export type HarissaChallenge = {
  question: string;
  answer: string;
};

export type ChachiaSpot = {
  id: number;
  name: string;
  lat: number;
  lng: number;
  image: string;
  rarity: string;
  xp: number;
  description: string;
  history: string;
  imageUrls: string[];
  harissa?: HarissaChallenge;
};

export const chachiaSpots: ChachiaSpot[] = [
  {
    id: 1,
    name: "Medina Gate",
    lat: 36.7975,
    lng: 10.1753,
    image: chachiaStandard,
    rarity: "Standard",
    xp: 15,
    description:
      "The main gate to the historic Medina of Tunis, a UNESCO World Heritage Site. The Medina is one of the best-preserved Islamic medieval cities in the world, featuring narrow winding streets, souks, mosques, and traditional hammams.",
    history:
      "The Medina of Tunis was founded in 698 CE and flourished under the Aghlabids, Fatimids, and later dynasties. Its gates served as crucial entry points and defensive structures. The main gate area has witnessed centuries of trade, culture, and daily life.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Medina_of_Tunis_6.jpg/800px-Medina_of_Tunis_6.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tunis_medina_3.jpg/800px-Tunis_medina_3.jpg",
    ],
  },
  {
    id: 3,
    name: "Sidi Bou Saïd",
    lat: 36.8733,
    lng: 10.3547,
    image: chachiaCultural,
    rarity: "Cultural",
    xp: 50,
    description:
      "A picturesque village perched on a cliff overlooking the Mediterranean, famous for its white and blue architecture, cobblestone streets, and artistic heritage. It has inspired many painters and writers, including Paul Klee and August Macke.",
    history:
      "Named after a 13th-century Sufi saint, Sidi Bou Saïd became an artists' colony in the early 20th century. Baron Rodolphe d'Erlanger promoted the blue-and-white color scheme in 1915. The village has preserved its Andalusian-inspired architecture.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Sidi_Bou_Said_Street_2.jpg/800px-Sidi_Bou_Said_Street_2.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Sidi_Bou_Said_%28Tunisia%29.jpg/800px-Sidi_Bou_Said_%28Tunisia%29.jpg",
    ],
    harissa: {
      question:
        "Harissa time! 🌶 In many Tunisian homes, I'm a spicy paste made from sun-dried chili peppers, garlic, and olive oil. People spread me on bread and add me to couscous. What am I?",
      answer: "harissa",
    },
  },
  {
    id: 5,
    name: "Cap Bon",
    lat: 36.8667,
    lng: 10.8,
    image: chachiaStandard,
    rarity: "Standard",
    xp: 15,
    description:
      "A fertile peninsula extending into the Mediterranean, known for its citrus groves, olive oil production, beaches, and historic sites. It offers stunning coastlines and traditional fishing villages.",
    history:
      "Cap Bon has been inhabited since ancient times. The Phoenicians and Romans valued its strategic location. Kerkouane, on the peninsula, is the only Punic city left untouched by Roman rebuilding. The region has long been an agricultural heartland.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Cap_Bon.jpg/800px-Cap_Bon.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Kerkouane_-_vue_g%C3%A9n%C3%A9rale.jpg/800px-Kerkouane_-_vue_g%C3%A9n%C3%A9rale.jpg",
    ],
  },
  {
    id: 6,
    name: "Sousse Medina",
    lat: 35.8256,
    lng: 10.6389,
    image: chachiaStandard,
    rarity: "Standard",
    xp: 15,
    description:
      "The historic walled Medina of Sousse, a UNESCO World Heritage Site. It features traditional architecture, souks, the Great Mosque, and the Ribat—a fortified monastery that served as both a religious and military structure.",
    history:
      "Sousse (ancient Hadrumetum) was founded by the Phoenicians. Under the Aghlabids in the 9th century, it became an important port and religious center. The Medina reflects typical early Islamic urban planning with its grid layout.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Sousse_Medina.jpg/800px-Sousse_Medina.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Sousse_aerial_view.jpg/800px-Sousse_aerial_view.jpg",
    ],
  },
  {
    id: 7,
    name: "Ribat of Sousse",
    lat: 35.8267,
    lng: 10.6412,
    image: chachiaLegendary,
    rarity: "Legendary",
    xp: 85,
    description:
      "One of the best-preserved ribats in the Maghreb, this fortified monastery served as a defensive outpost and religious retreat. Its imposing tower offers panoramic views of Sousse and the Mediterranean.",
    history:
      "The Ribat of Sousse was built in the 8th–9th centuries CE by the Aghlabid dynasty. It was part of a chain of coastal fortifications against Byzantine attacks. Monks-soldiers (murabitun) lived here, combining prayer with military duty.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Ribat_of_Sousse.jpg/800px-Ribat_of_Sousse.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Sousse_Ribat_tower.jpg/800px-Sousse_Ribat_tower.jpg",
    ],
    harissa: {
      question:
        "Harissa time! 🌊 The Ribat of Sousse has watched over traders and sailors for centuries. Which famous sea does it face? (Hint: it touches Tunisia, Italy, and Greece.)",
      answer: "mediterranean",
    },
  },
  {
    id: 8,
    name: "Sousse Great Mosque",
    lat: 35.8245,
    lng: 10.6378,
    image: chachiaCultural,
    rarity: "Cultural",
    xp: 45,
    description:
      "A historic mosque built in the 9th century, featuring Aghlabid architecture with its characteristic square minaret and fortress-like appearance. It remains an active place of worship and a key landmark of Sousse Medina.",
    history:
      "The Great Mosque of Sousse was constructed around 851 CE during the Aghlabid period. Its design reflects the defensive needs of the time, with thick walls and a minaret that also served as a watchtower. It exemplifies early Islamic architecture in North Africa.",
    imageUrls: [
      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Great_Mosque_of_Sousse.jpg/800px-Great_Mosque_of_Sousse.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Sousse_Mosque_minaret.jpg/800px-Sousse_Mosque_minaret.jpg",
    ],
  },
  {
    id: 10,
    name: "Sousse",
    lat: 53.3493,
    lng: -6.2605,
    image: chachiaStandard,
    rarity: "Standard",
    xp: 18,
    description:
      "The University of Sousse's engineering school (École Polytechnique de Sousse), a center of higher education and innovation. The campus represents Tunisia's commitment to technical education and research.",
    history:
      "Founded as part of Tunisia's expansion of higher education, Polytechnique Sousse trains engineers in various disciplines. It contributes to the region's development and connects Sousse's historic past with its modern aspirations.",
    imageUrls: [
      "https://www.polytecsousse.tn/wp-content/uploads/2025/05/campus-11.webp",
      "https://www.polytecsousse.tn/wp-content/uploads/2025/05/facade-poly-_2_.webp",
    ],
    harissa: {
      question:
        "Harissa time! 🎓 This school trains future engineers in a coastal Tunisian city famous for its Medina and beaches. Which city are we in?",
      answer: "sousse",
    },
  },
];

export const getChachiaSpotById = (id: number): ChachiaSpot | undefined =>
  chachiaSpots.find((spot) => spot.id === id);
