from tqdm import tqdm
import pandas as pd
import re


# Define amenity patterns and categories
AMENITY_PATTERNS = {
    "wifi": "\\bwifi\\b",
    "tv": "\\b(tv|hdtv)\\b",
    "air_conditioning": "\\b(air\\s?conditioning|ac)\\b",
    "heating": "\\b(heating|radiant heating|central heating)\\b",
    "washer": "\\b(washer|laundry)\\b",
    "dryer": "\\b(dryer)\\b",
    "kitchen": "\\b(kitchen|kitchenette)\\b",
    "oven": "\\boven\\b",
    "microwave": "\\b(micro\\s?wave|microwave)\\b",
    "bbq_grill": "\\b(bbq|grill)\\b",
    "fridge": "\\b(fridge|refrigerator|freezer)\\b",
    "fire_safety": "\\b(smoke alarm|fire extinguisher|carbon monoxide alarm|first aid kit)\\b",
    "security": "\\b(lock|safe|security|window guards|fireplace guards)\\b",
    "parking": "\\b(parking|garage|street parking|driveway)\\b",
    "pool": "\\b(pool)\\b",
    "hot_tub": "\\b(hot tub|jacuzzi)\\b",
    "sauna": "\\b(sauna)\\b",
    "gym": "\\b(gym|exercise|fitness)\\b",
    "children_friendly": "\\b(crib|pack\\s?n\\s?play|high chair|children|baby)\\b",
    "shower_equipment": "\\b(shower|shampoo|soap|conditioner|bathtub|bidet|hot water)\\b",
    "hair_dryer": "\\b(hair\\s?dryer|blow\\s?dryer)\\b",
    "workspace": "\\b(dedicated workspace|desk|laptop friendly|office)\\b",
    "pets_allowed": "\\b(pets allowed|pet-friendly)\\b",
    "outdoor_space": "\\b(patio|backyard|outdoor|terrace|balcony)\\b",
    "entertainment": "\\b(board games|arcade|pool table|gaming|game console|movie theater|bowling alley|life size games)\\b",
    "smart_lock": "\\b(smart\\s?lock|keypad|keyless|digital lock)\\b",
    "self_checkin": "\\b(self\\s?check-in|keypad|lockbox)\\b",
    "coffee_maker": "\\b(coffee maker|nespresso|keurig|espresso|coffee)\\b",
    "storage_space": "\\b(closet|dresser|wardrobe|storage)\\b",
    "scenic_view": "\\b(view|ocean view|mountain view|city skyline view|lake view|sea view|waterfront|beach view)\\b",
    "outdoor_seating": "\\b(outdoor furniture|outdoor seating|sun loungers|hammock)\\b",
    "ev_charger": "\\b(ev charger|electric vehicle charger)\\b",
    "baby_facilities": "\\b(crib|high chair|baby monitor|changing table|pack\\s?n\\s?play)\\b",
    "dishes_silverware": "\\b(dishes|silverware|cutlery|utensils)\\b",
    "cooking_basics": "\\b(cooking basics|oil|salt|pepper|baking sheet)\\b",
    "dishwasher": "\\b(dishwasher)\\b",
    "dining_table": "\\b(dining table|table)\\b",
    "wine_glasses": "\\b(wine glasses)\\b",
    "toaster": "\\b(toaster)\\b",
    "blender": "\\b(blender)\\b",
    "stove": "\\b(stove|gas stove|electric stove|induction stove)\\b",
    "rice_maker": "\\b(rice maker)\\b",
    "bread_maker": "\\b(bread maker)\\b",
    "bbq_utensils": "\\b(bbq utensils|barbecue utensils)\\b",
    "trash_compactor": "\\b(trash compactor)\\b",
    "extra_pillows_blankets": "\\b(extra pillows|blankets)\\b",
    "blackout_shades": "\\b(room-darkening shades|blackout curtains)\\b",
    "ceiling_fan": "\\b(ceiling fan)\\b",
    "portable_fan": "\\b(portable fan)\\b",
    "portable_heater": "\\b(portable heater)\\b",
    "cleaning_products": "\\b(cleaning products)\\b",
    "housekeeping": "\\b(cleaning available|housekeeping)\\b",
    "laundromat_nearby": "\\b(laundromat)\\b",
    "private_entrance": "\\b(private entrance)\\b",
    "elevator": "\\b(elevator)\\b",
    "single_level_home": "\\b(single level home)\\b",
    "window_guards": "\\b(window guards)\\b",
    "fireplace_guards": "\\b(fireplace guards)\\b",
    "outlet_covers": "\\b(outlet covers)\\b",
    "noise_monitors": "\\b(noise decibel monitors)\\b",
    "fire_pit": "\\b(fire pit)\\b",
    "bicycles": "\\b(bike|bikes)\\b",
    "kayak": "\\b(kayak)\\b",
    "ping_pong_table": "\\b(ping pong table)\\b",
    "sound_system": "\\b(sound system|bluetooth sound system|bose sound system|sonos sound system|record player|aux sound system)\\b",
    "piano": "\\b(piano)\\b",
    "beach_access": "\\b(beach access|beachfront|beach essentials|beach gear|beach towels|beach chairs)\\b",
    "shared_beach_access": "\\b(shared beach access)\\b",
    "lake_access": "\\b(lake access)\\b",
    "resort_access": "\\b(resort access|free resort access)\\b",
    "theme_room": "\\b(theme room)\\b",
    "portable_fans": "\\b(portable fan|portable fans)\\b",
    "breakfast": "\\b(breakfast)\\b",
    "babysitting": "\\b(babysitter recommendations|babysitting)\\b",
    "private_living_room": "\\b(private living room)\\b",
    "smoking_allowed": "\\b(smoking allowed)\\b",
    "mosquito_net": "\\b(mosquito net)\\b",
    "household_essentials": "\\b(essentials|hangers|iron|bed linens|linens|towels|bedding)\\b",
    "long_stay_friendly": "\\b(long term stays allowed|luggage dropoff allowed|extended stays)\\b",
    "kitchen_appliances": "\\b(hot water kettle|kettle)\\b",
    "indoor_entertainment": "\\b(books|reading material|library|book collection)\\b",
    "building_staff": "\\b(building staff|concierge)\\b",
    "internet_features": "\\b(ethernet connection|wired internet)\\b",
    "washer_dryer_features": "\\b(washer|laundry|dryer|drying rack|clothes drying rack)\\b",
    "indoor_fireplace": "\\b(indoor fireplace|fireplace|gas fireplace|wood-burning fireplace|electric fireplace)\\b",
    "host_greeting": "\\b(host greets you|host greets guests|host greets)\\b",
}
AMENITY_CATEGORIES = {
    "connectivity": ["wifi", "internet_features"],
    "entertainment": [
        "tv",
        "entertainment",
        "sound_system",
        "indoor_entertainment",
        "piano",
    ],
    "climate_control": [
        "air_conditioning",
        "heating",
        "ceiling_fan",
        "portable_fan",
        "portable_fans",
        "portable_heater",
    ],
    "kitchen": [
        "kitchen",
        "oven",
        "microwave",
        "bbq_grill",
        "fridge",
        "coffee_maker",
        "toaster",
        "blender",
        "stove",
        "rice_maker",
        "bread_maker",
        "bbq_utensils",
        "dishes_silverware",
        "cooking_basics",
        "dishwasher",
        "dining_table",
        "wine_glasses",
        "trash_compactor",
        "kitchen_appliances",
    ],
    "safety": [
        "fire_safety",
        "security",
        "window_guards",
        "fireplace_guards",
        "outlet_covers",
        "noise_monitors",
    ],
    "parking_transport": ["parking", "ev_charger"],
    "wellness": ["pool", "hot_tub", "sauna", "gym"],
    "family_friendly": ["children_friendly", "baby_facilities", "babysitting"],
    "bathroom": ["shower_equipment", "hair_dryer"],
    "workspace": ["workspace"],
    "pets": ["pets_allowed"],
    "outdoor": [
        "outdoor_space",
        "outdoor_seating",
        "fire_pit",
        "scenic_view",
        "beach_access",
        "lake_access",
        "resort_access",
        "bicycles",
        "kayak",
        "ping_pong_table",
        "shared_beach_access",
    ],
    "household": [
        "washer",
        "dryer",
        "cleaning_products",
        "housekeeping",
        "laundromat_nearby",
        "extra_pillows_blankets",
        "blackout_shades",
        "storage_space",
        "household_essentials",
        "long_stay_friendly",
        "washer_dryer_features",
    ],
    "accessibility": ["private_entrance", "elevator", "single_level_home"],
    "smart_home": ["smart_lock", "self_checkin"],
    "indoor_comfort": ["indoor_fireplace", "private_living_room"],
    "misc": [
        "theme_room",
        "breakfast",
        "smoking_allowed",
        "mosquito_net",
        "building_staff",
        "host_greeting",
    ],
}


def append_amenity_dummies(df, column, patterns, prefix="amenity_"):
    """
    Create dummy variables from an amenities column based on regex patterns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the amenities column.
        column (str): The column name containing the amenities as a list of strings.
        patterns (dict): A dictionary where keys are feature names and values are regex patterns.
        prefix (str): Prefix for the generated dummy columns.

    Returns:
        pd.DataFrame: The original DataFrame with new dummy columns added.
    """
    import warnings

    df = df.copy()  # Avoid modifying the original DataFrame

    # Explode the amenities column to have one amenity per row
    exploded_df = df[[column]].explode(column)

    # Initialize a DataFrame to store the dummy variables
    dummy_df = pd.DataFrame(index=df.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for amenity, regex in tqdm(patterns.items(), desc="Processing patterns"):
            # Create a boolean series for the regex match
            match_series = exploded_df[column].str.contains(
                regex, flags=re.IGNORECASE, regex=True, na=False
            )
            # Aggregate the match results back to the original dataframe's index
            dummy_df[f"{prefix}{amenity}"] = (
                match_series.groupby(exploded_df.index).any().astype(int)
            )  # Convert boolean to integer

    # Join the dummy variables back to the original dataframe
    df = df.join(dummy_df, rsuffix="_new")

    # Overwrite existing columns with new values if they exist
    for col in dummy_df.columns:
        if f"{col}_new" in df.columns:
            df[col] = df[f"{col}_new"]
            df.drop(columns=[f"{col}_new"], inplace=True)

    return df


def aggregate_amenity_categories(df, amenity_categories):
    """
    Computes category-based coverage and boolean presence columns for amenities efficiently.

    Parameters:
    df (pd.DataFrame): The DataFrame containing amenity columns prefixed with 'amenity_'.
    amenity_categories (dict): A dictionary where keys are category names and values are lists of amenities.

    Returns:
    pd.DataFrame: Updated DataFrame with new category columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    for category, amenities in amenity_categories.items():
        category_col = f"category_coverage_{category}"
        any_col = f"category_any_{category}"

        amenity_cols = [
            f"amenity_{amenity}"
            for amenity in amenities
            if f"amenity_{amenity}" in df.columns
        ]

        if amenity_cols:  # Ensure there are valid columns to sum
            df[category_col] = df[amenity_cols].sum(axis=1)
            df[any_col] = (df[category_col] > 0).astype(int)

    # Ensure a contiguous DataFrame for better performance
    return df


# Function to find matching patterns
def find_matching_patterns(amenity, patterns):
    matches = [
        key
        for key, pattern in patterns.items()
        if re.search(pattern, amenity, re.IGNORECASE)
    ]
    return matches
