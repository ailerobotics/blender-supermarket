"""Product catalog shared by the label generator (PIL) and the Blender builder.

All dimensions in metres: w = width across the shelf, d = depth into the shelf,
h = height. For round forms (can, bottle, jar, tub) w = d = diameter.
Brands are fictional. Colors are hex.

forms: box | bag | can | bottle | bottle_hh | jar | carton | tub | flat
"""

SKUS = [
    # ---------------- cereal & breakfast ----------------
    dict(id="cereal_choco",   name="Choco Crunch",   sub="chocolate rice puffs",   brand="Morning Mill", cat="cereal", form="box", w=0.19, d=0.06, h=0.30, bg="#4a1f0a", fg="#ffffff", accent="#f5b400", price=3.49, weight="500 g"),
    dict(id="cereal_corn",    name="Corn Flakes",    sub="golden toasted",         brand="Morning Mill", cat="cereal", form="box", w=0.19, d=0.06, h=0.30, bg="#f7c948", fg="#1a1a1a", accent="#c0392b", price=2.99, weight="500 g"),
    dict(id="cereal_honey",   name="Honey Loops",    sub="with real honey",        brand="Morning Mill", cat="cereal", form="box", w=0.24, d=0.08, h=0.33, bg="#e67e22", fg="#ffffff", accent="#f9e79f", price=4.29, weight="750 g"),
    dict(id="cereal_muesli",  name="Fruit Muesli",   sub="raisins • apple • nuts", brand="Valley Oats",  cat="cereal", form="bag", w=0.16, d=0.07, h=0.26, bg="#8e5a2b", fg="#ffffff", accent="#f1c40f", price=3.79, weight="750 g"),
    dict(id="cereal_oats",    name="Rolled Oats",    sub="whole grain",            brand="Valley Oats",  cat="cereal", form="box", w=0.14, d=0.08, h=0.24, bg="#2e86ab", fg="#ffffff", accent="#f0f3f4", price=1.99, weight="1 kg"),
    dict(id="cereal_granola", name="Crunchy Granola",sub="nut & seed clusters",    brand="Valley Oats",  cat="cereal", form="bag", w=0.15, d=0.07, h=0.24, bg="#2d5d34", fg="#ffffff", accent="#f5b400", price=4.49, weight="500 g"),
    # ---------------- coffee, tea, biscuits ----------------
    dict(id="coffee_ground",  name="Ground Coffee",  sub="medium roast",           brand="Kaffa Roasters", cat="coffee_tea", form="bag", w=0.12, d=0.06, h=0.20, bg="#3b1f0e", fg="#f5e6c8", accent="#c9a227", price=6.99, weight="500 g"),
    dict(id="coffee_beans",   name="Espresso Beans", sub="dark roast • arabica",   brand="Kaffa Roasters", cat="coffee_tea", form="bag", w=0.13, d=0.07, h=0.24, bg="#1a1a1a", fg="#ffffff", accent="#c0392b", price=8.99, weight="1 kg"),
    dict(id="coffee_instant", name="Instant Coffee", sub="rich & smooth",          brand="Kaffa Roasters", cat="coffee_tea", form="jar", w=0.075, d=0.075, h=0.16, bg="#6d4c41", fg="#ffffff", accent="#f5b400", price=5.49, weight="200 g"),
    dict(id="tea_black",      name="English Breakfast", sub="40 tea bags",         brand="Highgrove Tea",  cat="coffee_tea", form="box", w=0.14, d=0.07, h=0.08, bg="#8b0000", fg="#ffffff", accent="#f5b400", price=2.79, weight="125 g"),
    dict(id="tea_green",      name="Green Tea",      sub="25 tea bags • jasmine",  brand="Highgrove Tea",  cat="coffee_tea", form="box", w=0.14, d=0.07, h=0.08, bg="#2e7d32", fg="#ffffff", accent="#f0f3f4", price=2.49, weight="50 g"),
    dict(id="biscuit_digest", name="Digestive",      sub="wholewheat biscuits",    brand="Bakewell & Co", cat="coffee_tea", form="box", w=0.07, d=0.07, h=0.24, bg="#c8102e", fg="#ffffff", accent="#f5deb3", price=1.89, weight="400 g"),
    dict(id="biscuit_choc",   name="Choc Chip Cookies", sub="soft baked",          brand="Bakewell & Co", cat="coffee_tea", form="bag", w=0.14, d=0.05, h=0.20, bg="#3e2723", fg="#ffffff", accent="#f5b400", price=2.29, weight="200 g"),
    # ---------------- snacks & confectionery ----------------
    dict(id="chips_salt",     name="Sea Salt Chips", sub="kettle cooked",          brand="Crispo",   cat="snacks", form="bag", w=0.22, d=0.07, h=0.30, bg="#1565c0", fg="#ffffff", accent="#ffd54f", price=2.49, weight="150 g"),
    dict(id="chips_paprika",  name="Paprika Chips",  sub="smoky & sweet",          brand="Crispo",   cat="snacks", form="bag", w=0.22, d=0.07, h=0.30, bg="#c62828", fg="#ffffff", accent="#ffd54f", price=2.49, weight="150 g"),
    dict(id="chips_cheese",   name="Cheese & Onion", sub="crinkle cut",            brand="Crispo",   cat="snacks", form="bag", w=0.22, d=0.07, h=0.30, bg="#2e7d32", fg="#ffffff", accent="#ffd54f", price=2.49, weight="150 g"),
    dict(id="tortilla",       name="Tortilla Chips", sub="nacho cheese",           brand="El Sombrero", cat="snacks", form="bag", w=0.24, d=0.08, h=0.32, bg="#f57f17", fg="#1a1a1a", accent="#c62828", price=2.79, weight="200 g"),
    dict(id="pretzel",        name="Salted Pretzels",sub="crunchy sticks",         brand="Crispo",   cat="snacks", form="bag", w=0.18, d=0.06, h=0.26, bg="#6d4c41", fg="#ffffff", accent="#ffd54f", price=1.99, weight="175 g"),
    dict(id="popcorn",        name="Microwave Popcorn", sub="butter • 3 pack",     brand="PopTime",  cat="snacks", form="box", w=0.18, d=0.06, h=0.13, bg="#ffeb3b", fg="#1a1a1a", accent="#c62828", price=2.19, weight="3 x 90 g"),
    dict(id="choc_milk",      name="Milk Chocolate", sub="creamy alpine milk",     brand="Alpina",   cat="confectionery", form="flat", w=0.16, d=0.08, h=0.012, bg="#6a1b9a", fg="#ffffff", accent="#f5b400", price=1.49, weight="100 g"),
    dict(id="choc_dark",      name="Dark Chocolate", sub="70% cocoa",              brand="Alpina",   cat="confectionery", form="flat", w=0.16, d=0.08, h=0.012, bg="#212121", fg="#ffffff", accent="#c9a227", price=1.69, weight="100 g"),
    dict(id="candy_gummy",    name="Gummy Bears",    sub="fruit flavours",         brand="Sweetly",  cat="confectionery", form="bag", w=0.15, d=0.05, h=0.22, bg="#e91e63", fg="#ffffff", accent="#ffeb3b", price=1.79, weight="200 g"),
    dict(id="candy_mints",    name="Peppermints",    sub="extra strong",           brand="Sweetly",  cat="confectionery", form="box", w=0.09, d=0.04, h=0.14, bg="#0288d1", fg="#ffffff", accent="#ffffff", price=1.29, weight="150 g"),
    dict(id="cookies_box",    name="Butter Cookies", sub="danish style tin",       brand="Bakewell & Co", cat="confectionery", form="tub", w=0.18, d=0.18, h=0.07, bg="#1a237e", fg="#ffffff", accent="#c9a227", price=4.99, weight="454 g"),
    # ---------------- drinks ----------------
    dict(id="cola_can",       name="Cola",           sub="original taste",         brand="Vortex",   cat="soft_drinks", form="can", w=0.066, d=0.066, h=0.115, bg="#c62828", fg="#ffffff", accent="#ffffff", price=0.89, weight="330 ml"),
    dict(id="cola_zero_can",  name="Cola Zero",      sub="zero sugar",             brand="Vortex",   cat="soft_drinks", form="can", w=0.066, d=0.066, h=0.115, bg="#212121", fg="#ffffff", accent="#c62828", price=0.89, weight="330 ml"),
    dict(id="lemon_can",      name="Lemon Fizz",     sub="sparkling lemon",        brand="Vortex",   cat="soft_drinks", form="can", w=0.066, d=0.066, h=0.115, bg="#fbc02d", fg="#1a1a1a", accent="#2e7d32", price=0.89, weight="330 ml"),
    dict(id="orange_can",     name="Orange Burst",   sub="sparkling orange",       brand="Vortex",   cat="soft_drinks", form="can", w=0.066, d=0.066, h=0.115, bg="#ef6c00", fg="#ffffff", accent="#ffffff", price=0.89, weight="330 ml"),
    dict(id="cola_15l",       name="Cola",           sub="original taste",         brand="Vortex",   cat="soft_drinks", form="bottle", w=0.09, d=0.09, h=0.32, bg="#c62828", fg="#ffffff", accent="#ffffff", price=1.99, weight="1.5 L", liquid="#3a1a0a", cap="#c62828"),
    dict(id="lemon_15l",      name="Lemon Fizz",     sub="sparkling lemon",        brand="Vortex",   cat="soft_drinks", form="bottle", w=0.09, d=0.09, h=0.32, bg="#fbc02d", fg="#1a1a1a", accent="#2e7d32", price=1.99, weight="1.5 L", liquid="#e8f5c8", cap="#2e7d32"),
    dict(id="energy_can",     name="Volt Energy",    sub="taurine • caffeine",     brand="Volt",     cat="soft_drinks", form="can", w=0.053, d=0.053, h=0.168, bg="#1a1a1a", fg="#ffffff", accent="#00e676", price=1.49, weight="500 ml"),
    dict(id="water_15l",      name="Still Water",    sub="natural spring water",   brand="Aquella",  cat="water_juice", form="bottle", w=0.09, d=0.09, h=0.32, bg="#1e88e5", fg="#ffffff", accent="#ffffff", price=0.69, weight="1.5 L", liquid="#e9f4ff", cap="#1e88e5"),
    dict(id="water_sparkling",name="Sparkling Water",sub="lightly carbonated",     brand="Aquella",  cat="water_juice", form="bottle", w=0.09, d=0.09, h=0.32, bg="#00897b", fg="#ffffff", accent="#ffffff", price=0.79, weight="1.5 L", liquid="#e9f4ff", cap="#00897b"),
    dict(id="water_05l",      name="Still Water",    sub="natural spring water",   brand="Aquella",  cat="water_juice", form="bottle", w=0.066, d=0.066, h=0.22, bg="#1e88e5", fg="#ffffff", accent="#ffffff", price=0.49, weight="500 ml", liquid="#e9f4ff", cap="#1e88e5"),
    dict(id="juice_orange",   name="Orange Juice",   sub="100% pressed",           brand="SunGrove", cat="water_juice", form="carton", w=0.07, d=0.07, h=0.20, bg="#ff8f00", fg="#ffffff", accent="#ffffff", price=2.29, weight="1 L"),
    dict(id="juice_apple",    name="Apple Juice",    sub="cloudy • not from concentrate", brand="SunGrove", cat="water_juice", form="carton", w=0.07, d=0.07, h=0.20, bg="#7cb342", fg="#ffffff", accent="#ffffff", price=2.29, weight="1 L"),
    dict(id="juice_tropical", name="Tropical Mix",   sub="mango • pineapple",      brand="SunGrove", cat="water_juice", form="carton", w=0.07, d=0.07, h=0.20, bg="#fdd835", fg="#1a1a1a", accent="#e53935", price=2.49, weight="1 L"),
    # ---------------- canned & dry goods ----------------
    dict(id="can_tomato",     name="Chopped Tomatoes", sub="in rich tomato juice", brand="Casa Verde", cat="canned", form="can", w=0.074, d=0.074, h=0.11, bg="#c62828", fg="#ffffff", accent="#2e7d32", price=0.79, weight="400 g"),
    dict(id="can_beans",      name="Baked Beans",    sub="in tomato sauce",        brand="Casa Verde", cat="canned", form="can", w=0.074, d=0.074, h=0.11, bg="#1565c0", fg="#ffffff", accent="#f57f17", price=0.89, weight="415 g"),
    dict(id="can_corn",       name="Sweet Corn",     sub="whole kernel",           brand="Casa Verde", cat="canned", form="can", w=0.074, d=0.074, h=0.11, bg="#fdd835", fg="#1a1a1a", accent="#2e7d32", price=0.99, weight="340 g"),
    dict(id="can_chickpea",   name="Chickpeas",      sub="ready to eat",           brand="Casa Verde", cat="canned", form="can", w=0.074, d=0.074, h=0.11, bg="#f5deb3", fg="#1a1a1a", accent="#8d6e63", price=0.89, weight="400 g"),
    dict(id="can_tuna",       name="Tuna Chunks",    sub="in sunflower oil",       brand="Blue Fin",   cat="canned", form="can", w=0.085, d=0.085, h=0.04, bg="#0d47a1", fg="#ffffff", accent="#ffeb3b", price=1.39, weight="145 g"),
    dict(id="can_soup",       name="Tomato Soup",    sub="cream of tomato",        brand="Hearth",     cat="canned", form="can", w=0.074, d=0.074, h=0.11, bg="#b71c1c", fg="#ffffff", accent="#ffffff", price=1.19, weight="400 g"),
    dict(id="pasta_spag",     name="Spaghetti",      sub="durum wheat • bronze cut", brand="Nonna Rosa", cat="pasta_rice", form="box", w=0.08, d=0.045, h=0.27, bg="#1a237e", fg="#ffffff", accent="#ffeb3b", price=1.29, weight="500 g"),
    dict(id="pasta_penne",    name="Penne Rigate",   sub="durum wheat",            brand="Nonna Rosa", cat="pasta_rice", form="bag", w=0.13, d=0.06, h=0.24, bg="#1a237e", fg="#ffffff", accent="#ffeb3b", price=1.19, weight="500 g"),
    dict(id="pasta_fusilli",  name="Fusilli",        sub="tricolore",              brand="Nonna Rosa", cat="pasta_rice", form="bag", w=0.13, d=0.06, h=0.24, bg="#2e7d32", fg="#ffffff", accent="#ffeb3b", price=1.39, weight="500 g"),
    dict(id="rice_basmati",   name="Basmati Rice",   sub="extra long grain",       brand="Golden Field", cat="pasta_rice", form="bag", w=0.13, d=0.06, h=0.26, bg="#f9a825", fg="#1a1a1a", accent="#1a1a1a", price=2.99, weight="1 kg"),
    dict(id="rice_jasmine",   name="Jasmine Rice",   sub="fragrant thai",          brand="Golden Field", cat="pasta_rice", form="bag", w=0.13, d=0.06, h=0.26, bg="#00695c", fg="#ffffff", accent="#f9a825", price=2.79, weight="1 kg"),
    dict(id="sauce_tomato",   name="Tomato & Basil", sub="pasta sauce",            brand="Nonna Rosa", cat="pasta_rice", form="jar", w=0.075, d=0.075, h=0.14, bg="#c62828", fg="#ffffff", accent="#2e7d32", price=1.99, weight="400 g", lid="#c9a227", content="#b71c1c"),
    dict(id="sauce_pesto",    name="Pesto Genovese", sub="basil & pine nuts",      brand="Nonna Rosa", cat="pasta_rice", form="jar", w=0.065, d=0.065, h=0.10, bg="#2e7d32", fg="#ffffff", accent="#ffeb3b", price=2.49, weight="190 g", lid="#c9a227", content="#33691e"),
    # ---------------- baking, oils, condiments ----------------
    dict(id="flour",          name="Plain Flour",    sub="all purpose",            brand="Miller's",  cat="baking", form="bag", w=0.12, d=0.07, h=0.23, bg="#f5f5f5", fg="#1a1a1a", accent="#1565c0", price=0.99, weight="1 kg"),
    dict(id="sugar",          name="Caster Sugar",   sub="fine grain",             brand="Miller's",  cat="baking", form="bag", w=0.11, d=0.06, h=0.20, bg="#e3f2fd", fg="#1a1a1a", accent="#0d47a1", price=1.19, weight="1 kg"),
    dict(id="oil_olive",      name="Olive Oil",      sub="extra virgin",           brand="Casa Verde", cat="baking", form="bottle", w=0.075, d=0.075, h=0.29, bg="#2e7d32", fg="#ffffff", accent="#f9a825", price=6.49, weight="1 L", liquid="#c5b32a", cap="#1a1a1a"),
    dict(id="oil_sunflower",  name="Sunflower Oil",  sub="for frying & baking",    brand="Miller's",  cat="baking", form="bottle", w=0.08, d=0.08, h=0.29, bg="#fdd835", fg="#1a1a1a", accent="#e65100", price=2.49, weight="1 L", liquid="#f2d16b", cap="#fdd835"),
    dict(id="spice_paprika",  name="Paprika",        sub="sweet • ground",         brand="Spice Route", cat="baking", form="jar", w=0.045, d=0.045, h=0.10, bg="#bf360c", fg="#ffffff", accent="#ffeb3b", price=1.49, weight="45 g", lid="#212121", content="#c0392b"),
    dict(id="spice_pepper",   name="Black Pepper",   sub="whole peppercorns",      brand="Spice Route", cat="baking", form="jar", w=0.045, d=0.045, h=0.10, bg="#212121", fg="#ffffff", accent="#ffeb3b", price=1.79, weight="50 g", lid="#212121", content="#3e2723"),
    dict(id="ketchup",        name="Tomato Ketchup", sub="squeezy",                brand="Hearth",    cat="condiments", form="bottle_hh", w=0.07, d=0.05, h=0.24, bg="#c62828", fg="#ffffff", accent="#ffffff", price=1.89, weight="460 g", cap="#ffffff"),
    dict(id="mayo",           name="Mayonnaise",     sub="real egg",               brand="Hearth",    cat="condiments", form="jar", w=0.08, d=0.08, h=0.15, bg="#1565c0", fg="#ffffff", accent="#ffeb3b", price=2.19, weight="500 ml", lid="#1565c0", content="#fff8e1"),
    dict(id="mustard",        name="Dijon Mustard",  sub="strong",                 brand="Hearth",    cat="condiments", form="jar", w=0.06, d=0.06, h=0.11, bg="#f9a825", fg="#1a1a1a", accent="#1a1a1a", price=1.59, weight="200 g", lid="#f9a825", content="#e0b000"),
    dict(id="honey",          name="Clear Honey",    sub="pure blossom",           brand="Hive & Co", cat="condiments", form="jar", w=0.07, d=0.07, h=0.12, bg="#f9a825", fg="#1a1a1a", accent="#ffffff", price=3.49, weight="340 g", lid="#f9a825", content="#e6a100"),
    dict(id="jam_straw",      name="Strawberry Jam", sub="extra fruit",            brand="Hive & Co", cat="condiments", form="jar", w=0.07, d=0.07, h=0.11, bg="#c62828", fg="#ffffff", accent="#ffffff", price=2.29, weight="370 g", lid="#c62828", content="#8e0000"),
    dict(id="peanut_butter",  name="Peanut Butter",  sub="crunchy",                brand="Hive & Co", cat="condiments", form="jar", w=0.075, d=0.075, h=0.12, bg="#6d4c41", fg="#ffffff", accent="#f9a825", price=2.99, weight="340 g", lid="#6d4c41", content="#a1704a"),
    # ---------------- household ----------------
    dict(id="detergent",      name="Liquid Detergent", sub="colour care • 40 washes", brand="Brisk",  cat="cleaning", form="bottle_hh", w=0.13, d=0.08, h=0.30, bg="#1565c0", fg="#ffffff", accent="#00e5ff", price=7.99, weight="2 L", cap="#00bcd4"),
    dict(id="powder",         name="Washing Powder", sub="bio • 30 washes",        brand="Brisk",     cat="cleaning", form="box", w=0.20, d=0.08, h=0.28, bg="#1565c0", fg="#ffffff", accent="#ffeb3b", price=6.49, weight="1.95 kg"),
    dict(id="softener",       name="Fabric Softener",sub="spring fresh",           brand="Brisk",     cat="cleaning", form="bottle_hh", w=0.11, d=0.07, h=0.27, bg="#8e24aa", fg="#ffffff", accent="#f8bbd0", price=3.99, weight="1.5 L", cap="#ce93d8"),
    dict(id="dish_soap",      name="Dish Soap",      sub="lemon • ultra",          brand="Sparkle",   cat="cleaning", form="bottle_hh", w=0.065, d=0.05, h=0.23, bg="#fdd835", fg="#1a1a1a", accent="#2e7d32", price=1.49, weight="500 ml", cap="#2e7d32"),
    dict(id="spray_clean",    name="Multi-Surface Spray", sub="kills 99.9% germs", brand="Sparkle",   cat="cleaning", form="bottle_hh", w=0.09, d=0.06, h=0.28, bg="#00acc1", fg="#ffffff", accent="#ffffff", price=2.29, weight="750 ml", cap="#ffffff"),
    dict(id="bleach",         name="Thick Bleach",   sub="original",               brand="Sparkle",   cat="cleaning", form="bottle_hh", w=0.09, d=0.07, h=0.30, bg="#f5f5f5", fg="#1a1a1a", accent="#1565c0", price=1.29, weight="750 ml", cap="#1565c0"),
    dict(id="toilet_paper",   name="Toilet Tissue",  sub="3 ply • 9 rolls",        brand="Softline",  cat="paper", form="bag", w=0.33, d=0.11, h=0.33, bg="#f5f5f5", fg="#1a1a1a", accent="#4fc3f7", price=4.99, weight="9 rolls"),
    dict(id="kitchen_roll",   name="Kitchen Towel",  sub="2 ply • 4 rolls",        brand="Softline",  cat="paper", form="bag", w=0.23, d=0.115, h=0.26, bg="#ffffff", fg="#1a1a1a", accent="#ff7043", price=3.49, weight="4 rolls"),
    dict(id="tissues",        name="Facial Tissues", sub="soft • 100 sheets",      brand="Softline",  cat="paper", form="box", w=0.22, d=0.11, h=0.09, bg="#e1f5fe", fg="#1a1a1a", accent="#0288d1", price=1.29, weight="100 sheets"),
    dict(id="petfood_dog",    name="Dog Kibble",     sub="chicken & rice • adult", brand="Pawsome",   cat="pet", form="bag", w=0.28, d=0.10, h=0.42, bg="#e65100", fg="#ffffff", accent="#ffffff", price=12.99, weight="3 kg"),
    dict(id="petfood_cat",    name="Cat Food",       sub="salmon in jelly",        brand="Pawsome",   cat="pet", form="can", w=0.074, d=0.074, h=0.05, bg="#6a1b9a", fg="#ffffff", accent="#ffeb3b", price=0.79, weight="400 g"),
    # ---------------- personal care ----------------
    dict(id="shampoo",        name="Shampoo",        sub="argan oil • 2in1",       brand="Lumen",     cat="personal_care", form="bottle_hh", w=0.07, d=0.045, h=0.22, bg="#ad1457", fg="#ffffff", accent="#f8bbd0", price=3.29, weight="400 ml", cap="#ad1457"),
    dict(id="shower_gel",     name="Shower Gel",     sub="ocean fresh",            brand="Lumen",     cat="personal_care", form="bottle_hh", w=0.07, d=0.045, h=0.22, bg="#0277bd", fg="#ffffff", accent="#b3e5fc", price=2.49, weight="400 ml", cap="#0277bd"),
    dict(id="toothpaste",     name="Toothpaste",     sub="whitening • fluoride",   brand="Dentix",    cat="personal_care", form="box", w=0.045, d=0.04, h=0.19, bg="#ffffff", fg="#1a1a1a", accent="#0288d1", price=1.99, weight="75 ml"),
    dict(id="deodorant",      name="Deodorant",      sub="48h • aluminium free",   brand="Lumen",     cat="personal_care", form="can", w=0.05, d=0.05, h=0.15, bg="#37474f", fg="#ffffff", accent="#4dd0e1", price=2.79, weight="150 ml"),
    dict(id="soap_bar",       name="Soap Bar",       sub="milk & honey • 4 pack",  brand="Lumen",     cat="personal_care", form="box", w=0.19, d=0.05, h=0.09, bg="#fff8e1", fg="#1a1a1a", accent="#f9a825", price=2.19, weight="4 x 90 g"),
    dict(id="hand_wash",      name="Hand Wash",      sub="antibacterial",          brand="Lumen",     cat="personal_care", form="bottle_hh", w=0.065, d=0.05, h=0.18, bg="#00897b", fg="#ffffff", accent="#ffffff", price=1.79, weight="250 ml", cap="#ffffff"),
    # ---------------- dairy (chilled) ----------------
    dict(id="milk_whole",     name="Whole Milk",     sub="fresh • 3.5% fat",       brand="Meadowbrook", cat="dairy", form="carton", w=0.07, d=0.07, h=0.20, bg="#1565c0", fg="#ffffff", accent="#ffffff", price=1.09, weight="1 L"),
    dict(id="milk_semi",      name="Semi-Skimmed",   sub="fresh • 1.5% fat",       brand="Meadowbrook", cat="dairy", form="carton", w=0.07, d=0.07, h=0.20, bg="#2e7d32", fg="#ffffff", accent="#ffffff", price=1.09, weight="1 L"),
    dict(id="milk_oat",       name="Oat Drink",      sub="barista edition",        brand="Oatly-ish",   cat="dairy", form="carton", w=0.07, d=0.07, h=0.20, bg="#37474f", fg="#ffffff", accent="#ffb74d", price=1.99, weight="1 L"),
    dict(id="yogurt_straw",   name="Strawberry Yogurt", sub="creamy",              brand="Meadowbrook", cat="dairy", form="tub", w=0.10, d=0.10, h=0.09, bg="#e91e63", fg="#ffffff", accent="#ffffff", price=1.49, weight="500 g"),
    dict(id="yogurt_greek",   name="Greek Yogurt",   sub="10% fat",                brand="Meadowbrook", cat="dairy", form="tub", w=0.10, d=0.10, h=0.09, bg="#0d47a1", fg="#ffffff", accent="#ffffff", price=1.79, weight="500 g"),
    dict(id="butter",         name="Butter",         sub="salted",                 brand="Meadowbrook", cat="dairy", form="flat", w=0.10, d=0.065, h=0.04, bg="#f9a825", fg="#1a1a1a", accent="#ffffff", price=2.19, weight="250 g"),
    dict(id="cheese_cheddar", name="Mature Cheddar", sub="aged 12 months",         brand="Meadowbrook", cat="dairy", form="flat", w=0.12, d=0.075, h=0.035, bg="#e65100", fg="#ffffff", accent="#ffeb3b", price=3.49, weight="350 g"),
    dict(id="eggs",           name="Free Range Eggs",sub="large • 12",             brand="Meadowbrook", cat="dairy", form="box", w=0.25, d=0.11, h=0.075, bg="#8d6e63", fg="#ffffff", accent="#ffeb3b", price=2.99, weight="12 eggs"),
    # ---------------- beer & wine ----------------
    dict(id="beer_can",       name="Pilsner",        sub="premium lager • 5%",     brand="Nordbrau",  cat="beer_wine", form="can", w=0.066, d=0.066, h=0.168, bg="#1b5e20", fg="#ffffff", accent="#ffd54f", price=1.29, weight="500 ml"),
    dict(id="beer_bottle",    name="Pale Ale",       sub="hoppy • 5.5%",           brand="Nordbrau",  cat="beer_wine", form="bottle", w=0.065, d=0.065, h=0.23, bg="#e65100", fg="#ffffff", accent="#ffffff", price=1.79, weight="330 ml", liquid="#7a4a10", cap="#212121", glass="#3a2a10"),
    dict(id="wine_red",       name="Merlot",         sub="2023 • dry",             brand="Château Fable", cat="beer_wine", form="bottle", w=0.075, d=0.075, h=0.30, bg="#fdf6e3", fg="#4a0f0f", accent="#8b0000", price=7.99, weight="750 ml", liquid="#3b0a12", cap="#3b0a12", glass="#1f2a1a"),
    dict(id="wine_white",     name="Sauvignon Blanc",sub="2024 • crisp",           brand="Château Fable", cat="beer_wine", form="bottle", w=0.075, d=0.075, h=0.30, bg="#fdf6e3", fg="#2e4a1a", accent="#7cb342", price=7.49, weight="750 ml", liquid="#e8e3a0", cap="#c9a227", glass="#d9e8c8"),
    # ---------------- bakery ----------------
    dict(id="bread_white",    name="White Loaf",     sub="sliced • medium",        brand="Bakewell & Co", cat="bakery", form="bag", w=0.13, d=0.25, h=0.11, bg="#f5deb3", fg="#1a1a1a", accent="#c8102e", price=1.29, weight="800 g"),
    dict(id="bread_whole",    name="Wholemeal Loaf", sub="sliced • seeded",        brand="Bakewell & Co", cat="bakery", form="bag", w=0.13, d=0.25, h=0.11, bg="#8d6e63", fg="#ffffff", accent="#ffeb3b", price=1.49, weight="800 g"),
]

BY_ID = {s["id"]: s for s in SKUS}
CATEGORIES = {
    "cereal": "Cereal & Breakfast", "coffee_tea": "Coffee, Tea & Biscuits", "snacks": "Chips & Snacks",
    "confectionery": "Confectionery", "soft_drinks": "Soft Drinks", "water_juice": "Water & Juice",
    "canned": "Canned Goods", "pasta_rice": "Pasta, Rice & Sauces", "baking": "Baking & Oils",
    "condiments": "Condiments & Spreads", "cleaning": "Laundry & Cleaning", "paper": "Paper Goods",
    "pet": "Pet Food", "personal_care": "Personal Care", "dairy": "Dairy & Eggs", "beer_wine": "Beer & Wine",
    "bakery": "Bakery",
}

def by_cat(cat):
    return [s for s in SKUS if s["cat"] == cat]

if __name__ == "__main__":
    import collections
    c = collections.Counter(s["cat"] for s in SKUS)
    print(len(SKUS), "SKUs"); print(dict(c))
