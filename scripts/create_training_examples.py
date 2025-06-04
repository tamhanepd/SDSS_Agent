import json
import random

# Templates for training examples
fields = {
    "color": [("(p.u - p.g) > {val}", "u - g color greater than {val}"),
              ("(p.g - p.r) > {val}", "g - r color greater than {val}"),
              ("(p.r - p.i) > {val}", "r - i color greater than {val}"),
              ("(p.u - p.r) > {val}", "u - r color greater than {val}"),
              ("(p.u - p.i) > {val}", "u - i color greater than {val}"),
              ("(p.g - p.i) > {val}", "g - i color greater than {val}"),
              ("(p.u - p.g) < {val}", "u - g color less than {val}"),
              ("(p.g - p.r) < {val}", "g - r color less than {val}"),
              ("(p.r - p.i) < {val}", "r - i color less than {val}"),
              ("(p.u - p.r) < {val}", "u - r color less than {val}"),
              ("(p.u - p.i) < {val}", "u - i color less than {val}"),
              ("(p.g - p.i) < {val}", "g - i color less than {val}"),
              ("(p.g - p.r) BETWEEN {low} AND {high}", "g - r color between {low} and {high}")],
    "mag": [("p.u < {val}", "u-band magnitude less than {val}"),
            ("p.g < {val}", "g-band magnitude less than {val}"),
            ("p.r < {val}", "r-band magnitude less than {val}"),
            ("p.i < {val}", "i-band magnitude less than {val}"),
            ("p.z < {val}", "z-band magnitude less than {val}"),
            ("p.u > {val}", "u-band magnitude greater than {val}"),
            ("p.g > {val}", "g-band magnitude greater than {val}"),
            ("p.r > {val}", "r-band magnitude greater than {val}"),
            ("p.i > {val}", "i-band magnitude greater than {val}"),
            ("p.z > {val}", "z-band magnitude greater than {val}")],
    "redshift": [("s.z < {val}", "redshift less than {val}"),
                 ("s.z > {val}", "redshift greater than {val}"),
                 ("s.z BETWEEN {low} AND {high}", "redshift between {low} and {high}")],
    "type": [("p.type = 3", "photometric type 3")],
    "agn": [("(s.subClass = 'AGN' OR s.subClass = 'BROADLINE')", "AGN classification"),
            ("(s.subClass = 'AGN' OR s.subClass = 'BROADLINE')", "is classified as AGN")],
    "concentration": [("(p.petroR90_r / p.petroR50_r) > {val}", "concentration index above {val}"),
                      ("(p.petroR90_r / p.petroR50_r) < {val}", "concentration index below {val}")],
    "extinction": [("p.extinction_u < {val}", "u-band extinction less than {val}"),
                   ("p.extinction_g < {val}", "g-band extinction less than {val}"),
                   ("p.extinction_r < {val}", "r-band extinction less than {val}"),
                   ("p.extinction_u > {val}", "u-band extinction greater than {val}"),
                   ("p.extinction_g > {val}", "g-band extinction greater than {val}"),
                   ("p.extinction_r > {val}", "r-band extinction greater than {val}")],
    "spatial": [("p.objID IN (SELECT objID FROM dbo.fGetNearbyObjEq({ra}, {dec}, {rad}))",
                 "within {rad} degrees of RA={ra}, DEC={dec}"),
                ("p.ra BETWEEN {ra1} AND {ra2} AND p.dec BETWEEN {dec1} AND {dec2}",
                 "RA between {ra1} and {ra2}, and DEC between {dec1} and {dec2}")]
}

extra_columns_map = {
    "concentration": ["p.petroR90_r", "p.petroR50_r"],
    "extinction": ["p.extinction_g"],
    "type": ["p.type"],
    "agn": ["s.subClass"],
    "spatial": [],  # No extra output needed
    "color": [],    # Already includes u/g/r/i/z
    "mag": [],      # Already includes p.r, etc.
    "redshift": [], # Already includes s.z AS redshift
}

base_columns = [
    "p.objID", "p.ra", "p.dec", "s.z AS redshift",
    "p.u", "p.g", "p.r", "p.i", "p.z"
]

def generate_entry():
    used = random.sample(list(fields.keys()), random.randint(2, 5))
    condition_sql = []
    condition_nl = []
    selected_columns = base_columns[:]

    for key in used:
        field_sql, field_nl = random.choice(fields[key])

        if key == "redshift":
            if "BETWEEN" in field_sql:
                low = round(random.uniform(0.01, 0.2), 3)
                high = round(random.uniform(0.01, 0.2), 3)
                condition_sql.append(field_sql.format(low=min(low, high), high=max(low, high)))
                condition_nl.append(field_nl.format(low=min(low, high), high=max(low, high)))
            else:
                val = round(random.uniform(0.01, 0.3), 3)
                condition_sql.append(field_sql.format(val=val))
                condition_nl.append(field_nl.format(val=val))

        elif key == "color":
            if "BETWEEN" in field_sql:
                low = round(random.uniform(-2, 6), 3)
                high = round(random.uniform(-2, 6), 3)
                condition_sql.append(field_sql.format(low=min(low, high), high=max(low, high)))
                condition_nl.append(field_nl.format(low=min(low, high), high=max(low, high)))
            else:
                val = round(random.uniform(-1.0, 3.0), 2)
                condition_sql.append(field_sql.format(val=val))
                condition_nl.append(field_nl.format(val=val))

        elif key == "spatial":
            if "BETWEEN" in field_sql:
                ra1 = round(random.uniform(120, 220), 3)
                ra2 = round(ra1 + random.uniform(2, 5), 3)
                dec1 = round(random.uniform(-5, 30), 3)
                dec2 = round(dec1 + random.uniform(2, 5), 3)
                condition_sql.append(field_sql.format(ra1=ra1, ra2=ra2, dec1=dec1, dec2=dec2))
                condition_nl.append(field_nl.format(ra1=ra1, ra2=ra2, dec1=dec1, dec2=dec2))
            else:
                ra = round(random.uniform(120, 240), 3)
                dec = round(random.uniform(-10, 60), 3)
                rad = round(random.uniform(0.05, 0.3), 2)
                condition_sql.append(field_sql.format(ra=ra, dec=dec, rad=rad))
                condition_nl.append(field_nl.format(ra=ra, dec=dec, rad=rad))

        else:
            val = round(random.uniform(0.5, 20.0), 2)
            condition_sql.append(field_sql.format(val=val))
            condition_nl.append(field_nl.format(val=val))

        for col in extra_columns_map.get(key, []):
            if col not in selected_columns:
                selected_columns.append(col)

    prefix_templates = [
        "Find galaxies with {}.",
        "Which galaxies show {}?",
        "Select sources that satisfy {}.",
        "Give me galaxies having {}.",
        "Retrieve galaxies where {}.",
        "List galaxies matching {}.",
        "Show galaxies that meet {}."
    ]
    prefix = random.choice(prefix_templates)
    condition_description = ", ".join(condition_nl)
    instruction = prefix.format(condition_description)

    sql = (
        "SELECT TOP 10 " + ", ".join(selected_columns) +
        " FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestObjID = p.objID " +
        "WHERE " + " AND ".join(condition_sql)
    )

    return {"instruction": instruction, "output": sql}

if __name__ == "__main__":
    dataset = [generate_entry() for _ in range(5)]
    with open("training_examples_final.json", "w") as f:
        json.dump(dataset, f, indent=2)