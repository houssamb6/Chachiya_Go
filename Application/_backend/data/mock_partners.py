# Mock partner data per destination
# Replace with real partner data once partnerships are established

PARTNERS_DB = {
    "Sidi Bou Said": {
        "hotels": [
            {
                "name": "Dar Sidi Bou Said",
                "type": "Boutique Riad",
                "price_range": "$$",
                "highlight": "Stunning sea-view terrace, authentic Tunisian décor",
                "address": "Rue Sidi Bou Said, Sidi Bou Said"
            },
            {
                "name": "Hotel Dar Said",
                "type": "Heritage Hotel",
                "price_range": "$$$",
                "highlight": "Restored 19th-century palace, rooftop pool overlooking the bay",
                "address": "Rue Toumi, Sidi Bou Said"
            }
        ],
        "restaurants": [
            {
                "name": "Café des Nattes",
                "cuisine": "Tunisian café",
                "price_range": "$",
                "highlight": "Iconic hillside café, mint tea and makroudh since 1920",
                "must_try": "Mint tea with pine nuts"
            },
            {
                "name": "Au Bon Vieux Temps",
                "cuisine": "Traditional Tunisian",
                "price_range": "$$",
                "highlight": "Rooftop dining with panoramic sea views",
                "must_try": "Grilled fish and brik à l'oeuf"
            }
        ]
    },

    "Medina of Tunis": {
        "hotels": [
            {
                "name": "Dar Ben Gacem",
                "type": "Boutique Riad",
                "price_range": "$$",
                "highlight": "Hidden inside the medina, authentic architecture, rooftop terrace",
                "address": "Medina of Tunis"
            },
            {
                "name": "Hotel Majestic",
                "type": "Historic Hotel",
                "price_range": "$$",
                "highlight": "Art deco building from 1914, central location near Bab El Bhar",
                "address": "Avenue de Paris, Tunis"
            }
        ],
        "restaurants": [
            {
                "name": "Dar El Jeld",
                "cuisine": "Fine Tunisian",
                "price_range": "$$$",
                "highlight": "The most prestigious traditional restaurant in Tunis, inside a restored palace",
                "must_try": "Couscous royal and brick au thon"
            },
            {
                "name": "M'rabet",
                "cuisine": "Traditional Tunisian",
                "price_range": "$$",
                "highlight": "500-year-old café inside the souk, live Tunisian music on weekends",
                "must_try": "Lablabi (chickpea soup) and grilled merguez"
            }
        ]
    },

    "Djerba Island": {
        "hotels": [
            {
                "name": "Dar Dhiafa",
                "type": "Luxury Riad",
                "price_range": "$$$",
                "highlight": "Award-winning boutique hotel in a traditional Djerbian house with pool",
                "address": "Erriadh Village, Djerba"
            },
            {
                "name": "Hotel Lotos",
                "type": "Beach Resort",
                "price_range": "$$",
                "highlight": "Right on the beach, family-friendly, direct sea access",
                "address": "Zone Touristique, Djerba"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Baccar",
                "cuisine": "Fresh Seafood",
                "price_range": "$$",
                "highlight": "Best fresh catch on the island, fishermen bring their haul directly here",
                "must_try": "Grilled sea bream and octopus salad"
            },
            {
                "name": "Chez Slim",
                "cuisine": "Traditional Djerbian",
                "price_range": "$",
                "highlight": "Local favorite, no tourists — this is where Djerbans eat",
                "must_try": "Ojja (spicy egg and merguez stew)"
            }
        ]
    },

    "Sahara / Douz": {
        "hotels": [
            {
                "name": "Sahara Douz Camp",
                "type": "Desert Camp",
                "price_range": "$$",
                "highlight": "Sleep under the stars in a Bedouin tent surrounded by dunes",
                "address": "Douz Desert, 7km from town"
            },
            {
                "name": "Hotel Sahara Douz",
                "type": "Desert Hotel",
                "price_range": "$",
                "highlight": "Comfortable base camp, organizes all desert excursions",
                "address": "Avenue des Martyrs, Douz"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant El Mouradi",
                "cuisine": "Saharan Traditional",
                "price_range": "$",
                "highlight": "Authentic desert cooking — tagines slow-cooked in clay pots",
                "must_try": "Lamb tagine with dates and Saharan bread"
            },
            {
                "name": "Chez Hassan Camp Dinner",
                "cuisine": "Bedouin",
                "price_range": "$$",
                "highlight": "Dinner by firelight in the desert, live Bedouin music included",
                "must_try": "Full Bedouin dinner with couscous and desert tea"
            }
        ]
    },

    "Tozeur & Chebika": {
        "hotels": [
            {
                "name": "Dar Chahma",
                "type": "Boutique Hotel",
                "price_range": "$$",
                "highlight": "Traditional Tozeurian brick architecture, beautiful courtyard with palm trees",
                "address": "Medina of Tozeur"
            },
            {
                "name": "Ksar Bibi",
                "type": "Heritage Hotel",
                "price_range": "$$$",
                "highlight": "Converted fortified granary, unique architecture, desert views",
                "address": "Route de Nefta, Tozeur"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant La Palmeraie",
                "cuisine": "Oasis Cuisine",
                "price_range": "$$",
                "highlight": "Dining inside the palm grove, magical setting especially at night",
                "must_try": "Date-stuffed mechoui lamb and local deglet nour dates"
            },
            {
                "name": "Café de la République",
                "cuisine": "Tunisian Café",
                "price_range": "$",
                "highlight": "Old-school local café in the medina, unchanged since the 1960s",
                "must_try": "Tunisian coffee and zlebia pastry"
            }
        ]
    },

    "El Jem": {
        "hotels": [
            {
                "name": "Hotel Julius",
                "type": "City Hotel",
                "price_range": "$",
                "highlight": "Walking distance from the amphitheatre, rooftop view of the ruins",
                "address": "Avenue Hedi Chaker, El Jem"
            },
            {
                "name": "Dar El Jem",
                "type": "Guesthouse",
                "price_range": "$",
                "highlight": "Family-run guesthouse, home-cooked meals, warm local hospitality",
                "address": "Medina of El Jem"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Le Bonheur",
                "cuisine": "Tunisian",
                "price_range": "$",
                "highlight": "Best couscous in town, loved by locals and archaeologists alike",
                "must_try": "Friday couscous with lamb and vegetables"
            },
            {
                "name": "Café des Gladiateurs",
                "cuisine": "Café & Snacks",
                "price_range": "$",
                "highlight": "Terrace facing the amphitheatre — perfect for a coffee before your visit",
                "must_try": "Tunisian coffee and almond briouats"
            }
        ]
    },

    "Kairouan": {
        "hotels": [
            {
                "name": "Hotel Amina",
                "type": "City Hotel",
                "price_range": "$",
                "highlight": "Clean, central, 5 min walk from the Great Mosque",
                "address": "Avenue de la République, Kairouan"
            },
            {
                "name": "Dar Salam",
                "type": "Boutique Riad",
                "price_range": "$$",
                "highlight": "Restored medina house, tranquil courtyard, authentic atmosphere",
                "address": "Medina of Kairouan"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Sabra",
                "cuisine": "Traditional Kairouani",
                "price_range": "$",
                "highlight": "The place locals send their guests — honest, generous portions",
                "must_try": "Couscous with camel meat (a Kairouan specialty)"
            },
            {
                "name": "Patisserie Makroudh El Amel",
                "cuisine": "Pastry Shop",
                "price_range": "$",
                "highlight": "The most famous makroudh shop in Tunisia — buy a box to take home",
                "must_try": "Fresh makroudh with date paste and honey glaze"
            }
        ]
    },

    "Tabarka": {
        "hotels": [
            {
                "name": "Hotel Les Aiguilles",
                "type": "Boutique Hotel",
                "price_range": "$$",
                "highlight": "Facing the famous rock formations, steps from the beach",
                "address": "Avenue Habib Bourguiba, Tabarka"
            },
            {
                "name": "Dar Ismail Tabarka",
                "type": "Beach Resort",
                "price_range": "$$$",
                "highlight": "Private beach, diving center on-site, lush garden",
                "address": "Zone Touristique, Tabarka"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Le Corail",
                "cuisine": "Fresh Seafood",
                "price_range": "$$",
                "highlight": "Best seafood in Tabarka, the lobster is legendary",
                "must_try": "Grilled lobster and sea urchin starter"
            },
            {
                "name": "Café du Port",
                "cuisine": "Café & Seafood",
                "price_range": "$",
                "highlight": "Harbor-side café, fishermen's haul arrives here every morning",
                "must_try": "Fried calamari and local white wine"
            }
        ]
    },

    "Matmata & Tataouine": {
        "hotels": [
            {
                "name": "Hotel Sidi Driss",
                "type": "Troglodyte Hotel",
                "price_range": "$",
                "highlight": "The actual Star Wars Tatooine set — sleep where Luke Skywalker lived",
                "address": "Matmata"
            },
            {
                "name": "Hotel Diar Tataouine",
                "type": "Desert Hotel",
                "price_range": "$",
                "highlight": "Traditional ksar-style architecture, rooftop views of the south",
                "address": "Avenue Habib Bourguiba, Tataouine"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Matmata",
                "cuisine": "Berber Traditional",
                "price_range": "$",
                "highlight": "Underground dining room carved into the earth — unique experience",
                "must_try": "Berber tagine and khobz mella (ash bread)"
            },
            {
                "name": "Chez Abdallah",
                "cuisine": "Southern Tunisian",
                "price_range": "$",
                "highlight": "Family kitchen, grandmother's recipes, no menu — they cook what's fresh",
                "must_try": "Asida (semolina pudding) and slow-cooked lamb"
            }
        ]
    },

    "Hammamet": {
        "hotels": [
            {
                "name": "Hotel Sheraton Hammamet",
                "type": "Luxury Resort",
                "price_range": "$$$",
                "highlight": "Private beach, multiple pools, direct medina access",
                "address": "Zone Touristique, Hammamet"
            },
            {
                "name": "Dar Hayet",
                "type": "Boutique Hotel",
                "price_range": "$$",
                "highlight": "Charming guesthouse inside the old medina walls, rooftop sea views",
                "address": "Old Medina, Hammamet"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant La Bella Vista",
                "cuisine": "Tunisian & Mediterranean",
                "price_range": "$$",
                "highlight": "Terrace directly on the beach, sunset views are extraordinary",
                "must_try": "Seafood couscous and grilled dorade"
            },
            {
                "name": "Chez Achour",
                "cuisine": "Traditional Tunisian",
                "price_range": "$",
                "highlight": "A Hammamet institution since 1970 — locals eat here, not tourists",
                "must_try": "Brick à l'oeuf and lamb with olives"
            }
        ]
    },
    
    "Sousse": {
        "hotels": [
            {
                "name": "Hotel Tej Marhaba",
                "type": "Beach Resort",
                "price_range": "$$$",
                "highlight": "Beachfront location, multiple pools, close to Port El Kantaoui nightlife",
                "address": "Boulevard de la Corniche, Sousse"
            },
            {
                "name": "Dar El Medina Sousse",
                "type": "Boutique Riad",
                "price_range": "$$",
                "highlight": "Intimate riad inside the medina walls, rooftop terrace with sea breeze",
                "address": "Medina of Sousse"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Le Lido",
                "cuisine": "Tunisian & Seafood",
                "price_range": "$$",
                "highlight": "Terrace on the corniche, popular for sunset dinners and fresh grilled fish",
                "must_try": "Grilled sea bass and Sousse-style fish couscous"
            },
            {
                "name": "Bar & Grill Port El Kantaoui",
                "cuisine": "International & Tunisian",
                "price_range": "$$",
                "highlight": "Marina-side venue, transitions from dinner to live music after 10pm",
                "must_try": "Mixed grill platter and local Celtia beer"
            }
        ]
    },

    "Zaghouan": {
        "hotels": [
            {
                "name": "Hotel Les Sources",
                "type": "Nature Hotel",
                "price_range": "$",
                "highlight": "Surrounded by greenery near the springs, peaceful and refreshing",
                "address": "Route des Sources, Zaghouan"
            },
            {
                "name": "Dar Zaghouan",
                "type": "Guesthouse",
                "price_range": "$",
                "highlight": "Family-run guesthouse in the medina, home-cooked breakfasts with local produce",
                "address": "Medina of Zaghouan"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Ain El Kebira",
                "cuisine": "Traditional Tunisian",
                "price_range": "$",
                "highlight": "Open-air dining next to the natural springs, locals' favorite for weekend lunches",
                "must_try": "Lamb mechoui and seasonal vegetable salads"
            },
            {
                "name": "Café du Temple",
                "cuisine": "Tunisian Café",
                "price_range": "$",
                "highlight": "Charming café steps from the Roman Water Temple, great for a post-visit break",
                "must_try": "Tunisian coffee and date-filled pastries"
            }
        ]
    },

    "Ras Angela": {
        "hotels": [
            {
                "name": "Hotel Cap Angela",
                "type": "Coastal Guesthouse",
                "price_range": "$",
                "highlight": "Simple and clean with sea-facing rooms, ideal base for cliff walks and sunsets",
                "address": "Route du Cap, Bizerte"
            },
            {
                "name": "Hotel Nador Bizerte",
                "type": "City Hotel",
                "price_range": "$",
                "highlight": "Budget-friendly hotel in nearby Bizerte, 20 min drive to the cape",
                "address": "Avenue Habib Bourguiba, Bizerte"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant Le Phare",
                "cuisine": "Seafood & Grills",
                "price_range": "$",
                "highlight": "Right on the coast, fishermen drop their catch here daily — ultra-fresh",
                "must_try": "Fried red mullet and grilled prawns"
            },
            {
                "name": "Café Bord de Mer",
                "cuisine": "Café & Snacks",
                "price_range": "$",
                "highlight": "Cliffside terrace café, perfect for watching the Mediterranean at sunset",
                "must_try": "Tunisian mint tea and brik triangles"
            }
        ]
    },

    "Monastir Medina": {
        "hotels": [
            {
                "name": "Hotel Ribat Monastir",
                "type": "Heritage Hotel",
                "price_range": "$$",
                "highlight": "Steps from the Ribat fortress, rooftop with sea and medina views",
                "address": "Place de l'Indépendance, Monastir"
            },
            {
                "name": "Dar Monastir",
                "type": "Boutique Riad",
                "price_range": "$$",
                "highlight": "Restored traditional house inside the medina walls, serene courtyard garden",
                "address": "Medina of Monastir"
            }
        ],
        "restaurants": [
            {
                "name": "Restaurant La Marina",
                "cuisine": "Tunisian & Seafood",
                "price_range": "$$",
                "highlight": "Overlooking the marina, popular for long lunches with fresh fish and local wine",
                "must_try": "Grilled dorade and octopus à la Monastirienne"
            },
            {
                "name": "Café du Ribat",
                "cuisine": "Tunisian Café",
                "price_range": "$",
                "highlight": "Atmospheric café facing the fortress walls, great spot before or after the Ribat tour",
                "must_try": "Tunisian coffee with cardamom and almond briouats"
            }
        ]
    },
}