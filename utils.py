attribute_column_names = {
    "Attribute0": "checking_status",       # Status of existing checking account
    "Attribute1": "duration",              # Duration
    "Attribute2": "credit_history",        # Credit history
    "Attribute3": "purpose",               # Purpose
    "Attribute4": "credit_amount",         # Credit amount
    "Attribute5": "savings",               # Savings account/bonds
    "Attribute6": "employment",            # Present employment since
    "Attribute7": "installment_rate",      # Installment rate in percentage of disposable income
    "Attribute8": "personal_status_sex",   # Personal status and sex
    "Attribute9": "other_debtors",         # Other debtors / guarantors
    "Attribute10": "residence_since",      # Present residence since
    "Attribute11": "property",             # Property
    "Attribute12": "age",                  # Age
    "Attribute13": "installment_plans",    # Other installment plans
    "Attribute14": "housing",              # Housing
    "Attribute15": "existing_credits",     # Number of existing credits at this bank
    "Attribute16": "job",                  # Job
    "Attribute17": "num_dependents",       # Number of people being liable to provide maintenance for
    "Attribute18": "telephone",            # Telephone
    "Attribute19": "foreign_worker",       # Foreign worker
    "Attribute20": "risk"                  # 1 = Good, 2 = Bad
    }

# Define dictionaries for each qualitative attribute
attribute_mappings = {
    # Attribute 1: Status of existing checking account
    "A11": "< 0 DM",
    "A12": "0 <= ... < 200 DM",
    "A13": "... >= 200 DM / salary assignments for at least 1 year",
    "A14": "no checking account",
    
    # Attribute 3: Credit history
    "A30": "no credits taken/ all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account/ other credits existing (not at this bank)",
    
    # Attribute 4: Purpose
    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "(vacation - does not exist?)",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
    
    # Attribute 6: Savings account/bonds
    "A61": "... < 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": "... >= 1000 DM",
    "A65": "unknown/ no savings account",
    
    # Attribute 7: Present employment since
    "A71": "unemployed",
    "A72": "... < 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": "... >= 7 years",
    
    # Attribute 9: Personal status and sex
    "A91": "male: divorced/separated",
    "A92": "female: divorced/separated/married",
    "A93": "male: single",
    "A94": "male: married/widowed",
    "A95": "female: single",
    
    # Attribute 10: Other debtors / guarantors
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
    
    # Attribute 12: Property
    "A121": "real estate",
    "A122": "if not A121: building society savings agreement/ life insurance",
    "A123": "if not A121/A122: car or other, not in attribute 6",
    "A124": "unknown / no property",
    
    # Attribute 14: Other installment plans
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
    
    # Attribute 15: Housing
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
    
    # Attribute 17: Job
    "A171": "unemployed/ unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management/ self-employed/ highly qualified employee/ officer",
    
    # Attribute 19: Telephone
    "A191": "none",
    "A192": "yes, registered under the customer's name",
    
    # Attribute 20: Foreign worker
    "A201": "yes",
    "A202": "no",
}

print(attribute_column_names["Attribute0"])