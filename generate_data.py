import csv
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

items = [
("G001","Artisan Ceramic Tea Set",("200-300","handcrafted ceramic tea set birthday thank you thoughtful","tea lover artisan host"),"birthday,thank_you"),
("G002","Personalized Birth Flower Mug",("150-250","custom mug with recipient birth month flowers","tea lover coffee lover personalized"),"birthday,mothers_day"),
("G003","Handmade Scented Soy Candle",("100-200","lavender vanilla soy candle with reusable glass jar","spa lover cozy home"),"birthday,housewarming"),
("G004","Mini Succulent Gift Box",("150-250","low maintenance succulent collection in ceramic pots","plant lover student home"),"birthday,graduation"),
("G005","Leather Notebook Journal",("180-300","premium refillable journal for notes and ideas","student writer professional"),"graduation,birthday"),
("G006","Portable Espresso Maker",("350-500","compact travel espresso press for coffee enthusiasts","coffee lover traveler"),"birthday,travel"),
("G007","Cozy Knit Scarf",("200-350","soft warm scarf in neutral seasonal colors","fashion lover winter"),"christmas,holiday"),
("G008","Chocolate Truffle Hamper",("250-400","artisan assorted chocolates and cocoa treats","food lover hostess"),"birthday,thank_you"),
("G009","Wooden Puzzle Desk Game",("120-220","beautiful wooden brain teaser for quiet evenings","puzzle lover family"),"holiday,birthday"),
("G010","Custom Star Map Print",("200-350","personalized night sky illustration for a special date","romantic sentimental"),"anniversary,birthday"),
("G011","Smart Plant Sensor",("300-450","soil moisture and sunlight sensor for healthy plants","plant lover tech"),"housewarming,birthday"),
("G012","Bamboo Bath Spa Set",("250-380","bamboo tray soap candle and bath accessories","spa lover self care"),"mothers_day,thank_you"),
("G013","Reusable Travel Tumbler",("180-280","insulated stainless steel tumbler with spill resistant lid","traveler commuter"),"graduation,birthday"),
("G014","Handbound Sketchbook",("150-230","artist grade drawing pages in handmade cover","artist student"),"graduation,birthday"),
("G015","Board Game Night Set",("300-500","family strategy and card games in a gift box","family gamer"),"holiday,birthday"),
("G016","Sushi Making Kit",("250-400","beginner sushi kit with bamboo mat and recipe guide","food lover cook"),"birthday,housewarming"),
("G017","Indoor Herb Garden",("400-600","self watering indoor herb planter with seeds","cook plant lover"),"housewarming,birthday"),
("G018","Fitness Resistance Bands",("180-300","portable set for strength and stretching workouts","fitness enthusiast"),"birthday,graduation"),
("G019","Bluetooth Sleep Speaker",("350-550","compact speaker with calming sounds and timer","music lover wellness"),"birthday,thank_you"),
("G020","Embroidered Linen Handkerchief",("100-180","elegant personalized linen handkerchief set","sentimental traditional"),"wedding,anniversary"),
("G021","Digital Recipe Tablet Stand",("220-350","adjustable wooden stand for tablets and recipe cards","cook tech lover"),"mothers_day,birthday"),
("G022","Crystal Wine Glass Pair",("300-500","two hand blown glasses with gift box","wine lover host"),"anniversary,housewarming"),
("G023","Knitting Starter Kit",("220-360","colorful yarn needles and beginner pattern book","craft lover hobbyist"),"birthday,holiday"),
("G024","Travel Scratch Map",("180-300","gold foil world map for marking completed trips","traveler explorer"),"graduation,birthday"),
("G025","Gourmet Tea Sampler",("150-260","assorted loose leaf tea tins with brewing guide","tea lover host"),"birthday,thank_you"),
("G026","Wireless Charging Desk Lamp",("450-700","modern lamp with organizer and wireless charger","student professional"),"graduation,birthday"),
("G027","Personalized Puzzle Portrait",("250-400","custom jigsaw puzzle from a favorite photo","family sentimental"),"holiday,anniversary"),
("G028","Yoga Gift Bundle",("350-550","mat strap cork blocks and calming essential oils","fitness wellness"),"birthday,mothers_day"),
("G029","Retro Instant Camera",("700-900","instant film camera for memorable events","photographer traveler"),"graduation,birthday"),
("G030","Garden Tool Apron",("200-320","durable canvas apron with pockets for gardening tools","gardener home"),"mothers_day,housewarming"),
("G031","Mini Projector",("800-1000","portable cinema projector for movies and gaming","gamer movie lover"),"birthday,graduation"),
("G032","Custom Pet Portrait Pillow",("300-450","printed cushion from a custom illustration of a pet","pet lover sentimental"),"birthday,holiday"),
("G033","Ceramic Pour Over Coffee Set",("300-450","handcrafted dripper server and coffee filter papers","coffee lover artisan"),"birthday,thank_you"),
("G034","Calligraphy Starter Set",("150-250","pens ink and practice sheets for lettering","artist student"),"graduation,birthday"),
("G035","Foldable Picnic Blanket",("250-400","water resistant blanket with leather carrying strap","family traveler"),"summer,anniversary"),
("G036","Monthly Book Subscription",("300-450","curated books selected for the recipient reading taste","reader student"),"birthday,graduation"),
("G037","Bluetooth Thermal Printer",("400-600","pocket printer for photos notes and gift tags","student creative"),"graduation,birthday"),
("G038","Scented Bath Bomb Collection",("150-230","botanical bath bombs for relaxation and self care","spa lover teenager"),"birthday,thank_you"),
("G039","Handmade Woven Basket",("250-380","versatile storage basket made from natural fibers","artisan home lover"),"housewarming,mothers_day"),
("G040","Adventure Experience Voucher",("600-900","choice of kayaking hiking or cooking experience","adventurer traveler"),"birthday,anniversary"),
]

with (DATA / "items.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item_id","gift_name","price_range","occasion_tags","description","recipient_tags"])
    for item_id, name, text, recipient in items:
        # text tuple is (price, description, recipient); occasion tags stored separately
        occasion = dict((x[0], x[3]) for x in items).get(item_id, "")
        w.writerow([item_id,name,text[0],occasion,text[1],text[2]])

review_bank = {
 "positive": [
  "Thoughtful, beautiful and arrived with great quality.",
  "The recipient loved it and it felt personal.",
  "Excellent craftsmanship and perfect for the occasion.",
  "Useful, stylish and very good value.",
  "A wonderful gift that exceeded expectations."
 ],
 "neutral": [
  "Nice gift and reasonably easy to use.",
  "Good quality with an attractive presentation.",
  "It matched the description and arrived on time.",
  "A pleasant choice, though delivery could improve."
 ],
 "negative": [
  "Disappointing quality and smaller than expected.",
  "The recipient was not impressed and it felt overpriced.",
  "Average at best with weak packaging.",
  "Poor durability and not worth the price."
 ]
}

import random
random.seed(467)
ratings = []
for u in range(1,13):
    chosen = random.sample(range(40),24)
    for pos, idx in enumerate(chosen):
        rating = random.choice([3,4,4,5,5] if (u + idx) % 4 else [2,3,4])
        if rating >= 4:
            review = random.choice(review_bank["positive"])
        elif rating == 3:
            review = random.choice(review_bank["neutral"])
        else:
            review = random.choice(review_bank["negative"])
        ratings.append((f"U{u:02d}", items[idx][0], rating, review))

with (DATA / "ratings.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["user_id","item_id","rating","buyer_review"])
    w.writerows(ratings)

