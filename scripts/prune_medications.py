#!/usr/bin/env python3
"""
Create a pruned list of commonly prescribed medications from RxTerms with common dosages.
Target: ~1500-2000 entries covering the most common medications with their
most common strengths and dose forms, including both brand and generic names.

Strategy: Start with the full RxTerms list and keep medications based on:
1. Top ~500 most commonly prescribed generics
2. Common brand names
3. Most common dosages and forms
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "data" / "medications" / "rxterms_medications.json"
OUTPUT_PATH = ROOT_DIR / "data" / "medications" / "rxterms_medications_common.json"

# Expanded list of the top ~800 most commonly prescribed generics in the US
# Based on ClinCalc Top 300 + expanded list + research
TOP_GENERICS = [
    # Pain and inflammation
    "acetaminophen", "ibuprofen", "naproxen", "aspirin", "acetylsalicylic acid",
    "celecoxib", "diclofenac", "meloxicam", "indomethacin", "ketorolac",
    "piroxicam", "sulfasalazine", "hydrocodone", "oxycodone", "morphine",
    "hydromorphone", "codeine", "tramadol", "fentanyl", "methadone",
    "buprenorphine", "oxycodone/acetaminophen", "hydrocodone/acetaminophen",
    
    # Antibiotics
    "amoxicillin", "amoxicillin/clavulanate", "penicillin", "cephalexin", 
    "cefazolin", "ceftriaxone", "cefdinir", "cefixime", "cefepime",
    "ceftazidime", "azithromycin", "clarithromycin", "erythromycin",
    "doxycycline", "minocycline", "tetracycline", "clindamycin",
    "metronidazole", "flagyl", "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "ofloxacin", "sulfamethoxazole", "trimethoprim", "nitrofurantoin",
    "gentamicin", "tobramycin", "vancomycin", "linezolid", "daptomycin",
    "ceftaroline", "ceftobiprole", "telavancin", "fidaxomicin",
    
    # Cardiovascular
    "amlodipine", "lisinopril", "losartan", "valsartan", "telmisartan",
    "irbesartan", "olmesartan", "atenolol", "metoprolol", "propranolol",
    "carvedilol", "bisoprolol", "nebivolol", "sotalol", "diltiazem",
    "verapamil", "nifedipine", "nisoldipine", "felodipine", "isradipine",
    "hydralazine", "clonidine", "guanfacine", "prazosin", "doxazosin",
    "terazosin", "amethyst", "hydrochlorothiazide", "chlorthalidone",
    "furosemide", "bumetanide", "torsemide", "spironolactone", "eplerenone",
    "amlodipine/valsartan", "amlodipine/atorvastatin",
    "lisinopril/hydrochlorothiazide", "losartan/hydrochlorothiazide",
    "valsartan/hydrochlorothiazide", "telmisartan/hydrochlorothiazide",
    "atenolol/chlorthalidone", "bisoprolol/hydrochlorothiazide",
    
    # Cholesterol
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin",
    "fluvastatin", "pitavastatin", "ezetimibe", "fenofibrate", "gemfibrozil",
    "niacin", "omega-3", "fish oil", "prescription omega-3",
    
    # Diabetes
    "metformin", "glipizide", "glyburide", "glimepiride", "repaglinide",
    "nateglinide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "canagliflozin", "dapagliflozin", "empagliflozin", "ertugliflozin",
    "insulin", "insulin lispro", "insulin aspart", "insulin glargine",
    "insulin detemir", "insulin glulisine", "insulin npH", "insulin regular",
    "exenatide", "liraglutide", "semaglutide", "dulaglutide", "lixisenatide",
    
    # Mental health
    "fluoxetine", "sertraline", "paroxetine", "fluvoxamine", "citalopram",
    "escitalopram", "venlafaxine", "desvenlafaxine", "duloxetine", "milnacipran",
    "bupropion", "trazodone", "mirtazapine", "amitriptyline", "imipramine",
    "nortriptyline", "desipramine", "clomipramine", "doxepin",
    "alprazolam", "lorazepam", "diazepam", "clonazepam", "temazepam",
    "oxazepam", "chlorazepate", "triazolam", "midazolam",
    "quetiapine", "aripiprazole", "risperidone", "olanzapine", "ziprasidone",
    "paliperidone", "asenapine", "lurasidone", "cariprazine", "brexpiprazole",
    "haloperidol", "chlorpromazine", "thioridazine", "perphenazine",
    "fluphenazine", "prochlorperazine", "trifluoperazine",
    "lithium", "valproate", "valproic acid", "divalproex", "lamotrigine",
    "carbamazepine", "oxcarbazepine", "topiramate", "levetiracetam",
    "zonisamide", "ethosuximide", "gabapentin", "pregabalin", "clonazepam",
    "phenytoin", "phenobarbital", "primidone", "tiagabine", "vigabatrin",
    
    # Respiratory
    "albuterol", "levalbuterol", "ipratropium", "tiotropium", " formoterol",
    "salmeterol", "fluticasone", "budesonide", "mometasone", "beclomethasone",
    "ciclesonide", "flunisolide", "triamcinolone", "dexamethasone",
    "prednisone", "prednisolone", "methylprednisolone", "hydrocortisone",
    "theophylline", "aminophylline", "montelukast", "zileuton",
    "acetylcysteine", "benzonatate", "codeine/guaifenesin", "dextromethorphan",
    "guaifenesin", "pseudoephedrine", "phenylpropanolamine",
    
    # Gastrointestinal
    "omeprazole", "esomeprazole", "lansoprazole", "dexlansoprazole", "pantoprazole",
    "rabeprazole", "ranitidine", "famotidine", "cimetidine", "nizatidine",
    "sucralfate", "misoprostol", "ondansetron", "metoclopramide",
    "domperidone", "promethazine", "prochlorperazine", "dimenhydrinate",
    "meclizine", "diphenhydramine", "loratadine", "cetirizine", "fexofenadine",
    "ebastine", "desloratadine", "olopatadine",
    "simethicone", "docusate", "senna", "bisacodyl", "polyethylene glycol",
    "lactulose", "magnesium hydroxide", "magnesium citrate", "aluminum hydroxide",
    
    # Neurological
    "donepezil", "rivastigmine", "galantamine", "memantine", "rasagiline",
    "selegiline", "entacapone", "tolcapone", "pramipexole", "ropinirole",
    "rotigotine", "apomorphine", "carbidopa/levodopa", "amantadine",
    "baclofen", "cyclobenzaprine", "tizanidine", "dantrolene", "methocarbamol",
    "orphenadrine", "chlorzoxazone", "carisoprodol",
    "sumatriptan", "rizatriptan", "eletriptan", "zolmitriptan", "naratriptan",
    "frovatriptan", "almotriptan", "dihydroergotamine", "ergotamine",
    
    # Anticoagulants and antiplatelets
    "warfarin", "rivaroxaban", "apixaban", "dabigatran", "edoxaban",
    "heparin", "enoxaparin", "dalteparin", "tinzaparin", "fondaparinux",
    "clopidogrel", "ticagrelor", "prasugrel", "ticlopidine", "aspirin/dipyridamole",
    
    # Hormones and endocrine
    "levothyroxine", "liothyronine", "thyroid", "methimazole", "propylthiouracil",
    "estradiol", "estrogen", "progesterone", "medroxyprogesterone", "norethindrone",
    "ethinyl estradiol", "drospirenone", "levonorgestrel", "norgestimate",
    "testosterone", "methyltestosterone", "oxandrolone", "anastrozole",
    "letrozole", "exemestane", "tamoxifen", "raloxifene", "clomiphene",
    "leuprolide", "goserelin", "nafarelin", "histrelin", "degarelix",
    "octreotide", "lanreotide", "pasireotide",
    
    # Immunosuppressants
    "tacrolimus", "cyclosporine", "sirolimus", "everolimus", "mycophenolate",
    "azathioprine", "mercaptopurine", "methotrexate", "leflunomide",
    "tofacitinib", "baricitinib", "upadacitinib", "filgotinib",
    
    # Antivirals
    "acyclovir", "valacyclovir", "famciclovir", "penciclovir", "ganciclovir",
    "valganciclovir", "foscarnet", "cidofovir", "oseltamivir", "zanamivir",
    "peramivir", "baloxavir", "rimantadine", "amantadine",
    "lamivudine", "emtricitabine", "tenofovir", "abacavir", "zidovudine",
    "nevirapine", "efavirenz", "rilpivirine", "doravirine", "etravirine",
    "dolutegravir", "raltegravir", "elvitegravir", "bictegravir",
    "cobicistat", "ritonavir", "lopinavir", "atazanavir", "darunavir",
    "fosamprenavir", "tipranavir", "enfuvirtide", "maraviroc",
    
    # Antifungals
    "fluconazole", "ketoconazole", "itraconazole", "voriconazole", "posaconazole",
    "isavuconazole", "terbinafine", "griseofulvin", "nystatin", "amphotericin",
    "amphotericin b", "candicidin", "flucytosine", "capsofungin", "micafungin",
    "anidulafungin",
    
    # Eye and ear
    "latanoprost", "bimatoprost", "travoprost", "tafluprost", "unoprostone",
    "brimonidine", "apraclonidine", "dapiprazole", "timolol", "betaxolol",
    "levobunolol", "carteolol", "metipranolol", "brinzolamide", "dorzolamide",
    "acetazolamide", "pilocarpine", "carbachol", "physostigmine",
    "atropine", "tropicamide", "cyclopentolate", "homatropine",
    "ciprofloxacin ophthalmic", "ofloxacin ophthalmic", "moxifloxacin ophthalmic",
    "gatifloxacin ophthalmic", "besifloxacin ophthalmic",
    "tobramycin ophthalmic", "gentamicin ophthalmic",
    "prednisolone acetate", "loteprednol", "fluorometholone",
    "diclofenac ophthalmic", "ketorolac ophthalmic", "bromfenac ophthalmic",
    "nepafenac ophthalmic", "flurbiprofen ophthalmic",
    
    # Skin
    "hydrocortisone topical", "triamcinolone topical", "betamethasone topical",
    "fluocinolone topical", "mometasone topical", "clobetasol topical",
    "desonide topical", "fluocinonide topical", "halobetasol topical",
    "tacrolimus topical", "pimecrolimus topical", "calcium",
    "retinoic acid", "tretinoin", "adapalene", "tazarotene",
    "benzoyl peroxide", "salicylic acid", "sulfur", "resorcinol",
    "clindamycin topical", "erythromycin topical", "metronidazole topical",
    "dapsone topical", "azelaic acid", "imiquimod", "podofilox",
    "5-fluorouracil", "masoprocol", "mechlorethamine",
    
    # Vitamins and supplements
    "vitamin a", "vitamin b1", "vitamin b2", "vitamin b3", "vitamin b6", 
    "vitamin b12", "vitamin c", "vitamin d", "vitamin d2", "vitamin d3",
    "vitamin e", "vitamin k", "folic acid", "folate", "biotin",
    "calcium", "magnesium", "potassium", "iron", "zinc",
    "copper", "selenium", "iodine", "chromium", "manganese",
    "molybdenum", "coenzyme q10", "alpha lipoic acid", "omega-3",
    "fish oil", "flaxseed oil", "borage oil", "evening primrose oil",
    "probiotic", "prebiotic", "multivitamin",
    
    # Cancer
    "capecitabine", "fluorouracil", "gemcitabine", "cytarabine", "methotrexate",
    "pemetrexed", "mercaptopurine", "thioguanine", "fludarabine", "cladribine",
    "pentostatin", "nelarabine", "bendamustine", "temozolomide", "procarbazine",
    "dacarbazine", "cisplatin", "carboplatin", "oxaliplatin",
    "cyclophosphamide", "ifosfamide", "tropfosfamide", "mechlorethamine",
    "chlorambucil", "melphalan", "busulfan", "thiotepa",
    "imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib",
    "erlotinib", "gefitinib", "afatinib", "osimertinib", "dacomitinib",
    "crizotinib", "alectinib", "brigatinib", "lorlatinib",
    "trastuzumab", "pertuzumab", "ado-trastuzumab emtansine",
    "bevacizumab", "cetuximab", "panitumumab", "ramucirumab",
    "rituximab", "ofatumumab", "obinutuzumab", "ibritumomab", "tositumomab",
    "ipilimumab", "nivolumab", "pembrolizumab", "atezolizumab", "avelumab",
    "durvalumab",
    
    # Urinary
    "sildenafil", "tadalafil", "vardenafil", "avanafil", "alprostadil",
    "testosterone topical", "testosterone gel", "testosterone patch",
    "testosterone injection", "phenazopyridine", "oxybutynin",
    "tolterodine", "solifenacin", "darifenacin", "trospium", "fesoterodine",
    "mirabegron", "flavoxate", "bethanechol", "neostigmine", "pyridostigmine",
    "tamsulosin", "alfuzosin", "silodosin", "doxazosin", "terazosin",
    "finasteride", "dutasteride", "minoxidil",
    
    # Miscellaneous
    "colchicine", "allopurinol", "febuxostat", "probenecid", "sulfinpyrazone",
    "gout medication", "podiatry", "osteoporosis", "bisphosphonate",
    "alendronate", "risedronate", "zoledronic acid", "ibandronate",
    "denosumab", "raloxifene", "bazedoxifene", "teriparatide",
    "calcitonin", "ergocalciferol", "cholecalciferol", "calcifediol",
    "paricalcitol", "doxercalciferol", "calcitriol",
]

# Common brand names that should be included
# Many of these will be caught by the generic list, but some need explicit inclusion
COMMON_BRANDS = [
    "Tylenol", "Advil", "Motrin", "Aleve", "Aspirin", "Bayer", "Excedrin",
    "Amoxil", "Augmentin", "Keflex", "Cefzil", "Omnicef", "Suprax", "Rocephin",
    "Zithromax", "Z-Pak", "Biaxin", "Erythrocin", "E-Mycin", "Vibramycin", 
    "Doryx", "Minocin", "Cleocin", "Flagyl", "Cipro", "Levaquin", "Avelox",
    "Noroxin", "Floxin", "Bactrim", "Septra", "Macrobid", "Macrodantin",
    "Norvasc", "Zestril", "Prinivil", "Cozaar", "Diovan", "Micardis", "Avapro",
    "Benicar", "Lopressor", "Toprol", "Inderal", "Coreg", "Zebeta", "Bystolic",
    "Betapace", "Calan", "Verelan", "Cardizem", "Procardia", "Adalat", "Sular",
    "Catapres", "Intuniv", "Minipress", "Cardura", "Hytrin", "Capoten", 
    "Lopid", "Zocor", "Lipitor", "Crestor", "Pravachol", "Mevacor", "Lescol", 
    "Vytorin", "Tricor", "Lopid", "Niaspan", "Lovaza", "Vascepa",
    "Glucophage", "Glucotrol", "DiaBeta", "Micronase", "Glynase", "Amaryl",
    "Actos", "Avandia", "Precose", "Glyset", "Januvia", "Onglyza", "Tradjenta",
    "Nesina", "Invokana", "Farxiga", "Jardiance", "Steglatro", "Humalog", 
    "NovoLog", "Apidra", "Lantus", "Levemir", "Toujeo", "Basaglar", "Byetta",
    "Victoza", "Trulicity", "Ozempic", "Bydureon", "Adlyxin",
    "Prozac", "Zoloft", "Paxil", "Celexa", "Lexapro", "Effexor", "Pristiq",
    "Cymbalta", "Savella", "Wellbutrin", "Desyrel", "Remeron", "Elavil", 
    "Tofranil", "Norpramin", "Pamelor", "Sinequan", "Silence", "Xanax", 
    "Ativan", "Valium", "Klonopin", "Tranxene", "Librium", "Serax", 
    "Halcion", "Restoril", "Seroquel", "Abilify", "Risperdal", "Zyprexa", 
    "Geodon", "Invega", "Saphris", "Latuda", "Vraylar", "Rexulti",
    "Haldol", "Thorazine", "Mellaril", "Prolixin", "Compazine", "Phenergan",
    "Lithobid", "Eskalith", "Depakote", "Depakene", "Lamictal", "Tegretol",
    "Trileptal", "Topamax", "Keppra", "Zonegran", "Dilantin", "Phenobarbital",
    "Mysoline", "Neurontin", "Lyrica", "Klonopin", "Onfi",
    "Proventil", "Ventolin", "ProAir", "Atrovent", "Spiriva", "Foradil", 
    "Serevent", "Flovent", "Pulmicort", "Asmanex", "Nasonex", "Flonase",
    "Veramyst", "Rhinocort", "Qvar", "Aerospan", "Combivent", "Duoneb",
    "Advair", "Symbicort", "Breo", "Anoro", "Stiolto", "Bevespi",
    "Prilosec", "Nexium", "Prevacid", "Protonix", "Dexilant", "Aciphex",
    "Zantac", "Pepcid", "Tagamet", "Axid", "Carafate", "Cytotec", "Zofran",
    "Reglan", "Compazine", "Phenergan", "Dramamine", "Bonine", "Antivert",
    "Miralax", "Colace", "Senokot", "Dulcolax", "Milk of Magnesia", 
    "Mylanta", "Maalox", "Tums", "Rolaids", "Pepto-Bismol", "Gas-X",
    "Aricept", "Namenda", "Exelon", "Razadyne", "Azilect", "Eldepryl",
    "Comtan", "Tasmar", "Mirapex", "Requip", "Neupro", "Apokyn",
    "Sinemet", "Parlodel", "Symmetrel", "Imitrex", "Maxalt", "Zomig",
    "Relpax", "Axert", "Frova", "Migranal",
    "Coumadin", "Jantoven", "Xarelto", "Eliquis", "Pradaxa", "Savaysa",
    "Heparin", "Lovenox", "Fragmin", "Innohep", "Arixtra", "Plavix",
    "Brilinta", "Effient", "Ticlid", "Aggrenox",
    "Synthroid", "Levothroid", "Levoxyl", "Unithroid", "Armour Thyroid",
    "Tapazole", "PTU", "Estrace", "Premarin", "Provera", "Aygestin",
    "Prometrium", "Ortho Tri-Cyclen", "Yasmin", "Lo Loestrin", "Seasonale",
    "AndroGel", "Testim", "Axiron", "Fortesta", "Androderm", "Striant",
    "Testopel", "Arimidex", "Femara", "Aromasin", "Nolvadex", "Evista",
    "Clomid", "Serophene", "Lupron", "Zoladex", "Synarel", "Vantas",
    "Sandostatin", "Somavert",
    "Prograf", "Neoral", "Sandimmune", "Rapamune", "CellCept", 
    "Imuran", "Purinethol", "Rheumatrex", "Trexall", "Xeljanz", "Olumiant",
    "Rinvoq", "Zovirax", "Valtrex", "Famvir", "Denavir", "Tamiflu", 
    "Relenza", "Rapivab", "Symphony", "Flumadine",
    "Epivir", "Ziagen", "Viread", "Truvada", "Atripla", "Combivir", 
    "Trizivir", "Stribild", "Genvoya", "Biktarvy", "Dovato", "Juluca",
    "Descovy", "Odefsey", "Triumeq", "Tivicay", "Isentress", "Selzentry",
    "Fuzeon", "Celsentri", "Diflucan", "Nizoral", "Sporanox", "Vfend",
    "Noxafil", "Cresemba", "Lamisil", "Grifulvin", "Nystop", "Mycostatin",
    "AmBisome", "Fungizone", "Cancidas", "Mycoamine", "Eraxis",
    "Xalatan", "Lumigan", "Travatan", "Rescula", "Alphagan", "Lopidine",
    "Trusopt", "Cosopt", "Azopt", "Diamox", "Pilocarpine", "Isopto Carpine",
    "Ocupress", "Ciloxan", "Ocuflox", "Vigamox", "Besivance", "Tobrex",
    "Garamycin", "Pred Forte", "Lotemax", "Alrex", "FML", "Voltarol",
    "Acular", "Xalatan", "Patanol", "Pataday", "Bepreve", "Lastacaft",
    "Cortef", "Hydrocortone", " Kenalog", "Aristocort", "Cordran", 
    "Lidex", "Luxiq", "Clobex", "Temovate", "Elocon", "Retin-A", 
    "Renova", "Differin", "Tazorac", "Benzaclin", "Epiduo", "Ziana",
    "Cleocin T", "Erygel", "MetroGel", "Aczone", "Soriatane",
    "Carac", "Efudex", "Aldara", "Condylox", "Valtrex",
    "Onexton", "Duac", "Epiduo", "Acanya", "BenzaClin",
]


def should_include(item):
    """Check if medication should be included based on generic, brand, or common names."""
    label = item.get('label', '').casefold()
    display = item.get('display_name', '').casefold()
    synonym = item.get('synonym', '').casefold()
    
    # Check all combined names
    all_names = [label, display, synonym]
    
    # Check against top generics
    for g in TOP_GENERICS:
        for name in all_names:
            if g.casefold() in name:
                return True
    
    # Check against common brands
    for brand in COMMON_BRANDS:
        for name in all_names:
            if brand.casefold() in name:
                return True
    
    return False


def main():
    if not INDEX_PATH.exists():
        print(f"Error: {INDEX_PATH} not found")
        return 1
    
    with open(INDEX_PATH) as f:
        data = json.load(f)
    
    print(f"Processing {len(data['medications'])} total medications...")
    
    # Collect all matching items
    matching_items = [item for item in data['medications'] if should_include(item)]
    print(f"Found {len(matching_items)} matching medications")
    
    # Sort by priority: prefer oral forms, then by label
    form_priority = {
        'oral pill': 0, 'oral capsule': 0, 'oral tablet': 0, 'capsule': 0, 'tablet': 0,
        'oral liquid': 1, 'oral suspension': 1, 'oral solution': 1, 'suspension': 1, 'solution': 1,
        'oral': 2, 'oral disintegrating': 2, 'oral delayed release': 2, 'oral extended release': 2,
        'chewable': 2, 'disintegrating': 2, 'extended release': 2, 'delayed release': 2,
        'topical': 3, 'cream': 3, 'ointment': 3, 'gel': 3, 'lotion': 3, 'spray': 3,
        'inhalation': 4, 'inhalant': 4, 'aerosol': 4, 'metered dose': 4,
        'nasal': 4, 'nasal spray': 4,
        'injectable': 5, 'intravenous': 5, 'subcutaneous': 5, 'injection': 5,
        'rectal': 6, 'suppository': 6,
        'ophthalmic': 7, 'eye': 7, 'otic': 7, 'ear': 7,
    }
    
    def sort_key(item):
        dose_form = item.get('dose_form', '').casefold()
        # Check both the dose_form and try to extract from label
        if not dose_form or dose_form not in form_priority:
            label_lower = item.get('label', '').casefold()
            for form, priority in form_priority.items():
                if form in label_lower:
                    dose_form = form
                    break
        
        priority = form_priority.get(dose_form, 10)
        # Prefer items with strength (have dosages)
        has_strength = 0 if item.get('strength') else 1
        return (priority, has_strength, item['label'].casefold())
    
    sorted_items = sorted(matching_items, key=sort_key)
    
    # Take items until we reach ~2000, deduplicate by label
    pruned = []
    seen_labels = set()
    
    for item in sorted_items:
        label_lower = item['label'].casefold()
        if label_lower not in seen_labels:
            pruned.append(item)
            seen_labels.add(label_lower)
        if len(pruned) >= 2000:
            break
    
    print(f"After deduplication: {len(pruned)} unique medications")
    
    # Add common supplements that might not be in RxTerms
    COMMON_SUPPLEMENTS = [
        "biotin", "calcium carbonate", "calcium citrate", "calcium with vitamin d",
        "coenzyme q10", "fish oil", "flaxseed oil", "folic acid",
        "iron", "magnesium oxide", "magnesium citrate", "magnesium glycinate",
        "melatonin", "multivitamin", "multivitamin with minerals", "omega-3",
        "omega-3 fish oil", "potassium chloride", "probiotic", "selenium",
        "vitamin a", "vitamin b complex", "vitamin b1", "vitamin b12",
        "vitamin b6", "vitamin c", "vitamin d", "vitamin d2", "vitamin d3",
        "vitamin e", "vitamin k", "zinc",
    ]
    
    for supp in COMMON_SUPPLEMENTS:
        supp_lower = supp.casefold()
        found = any(supp_lower in item['label'].casefold() for item in pruned)
        if not found:
            pruned.append({
                "label": supp,
                "rxcui": "",
                "display_name": supp,
                "strength": "",
                "route": "Oral",
                "dose_form": "Supplement",
                "synonym": ""
            })
    
    pruned.sort(key=lambda x: x['label'].casefold())
    
    output = {
        "source": "RxTerms (pruned to common medications with dosages and brand names)",
        "source_url": data.get("source_url", ""),
        "release": data.get("release", ""),
        "generated_on": data.get("generated_on", ""),
        "count": len(pruned),
        "medications": pruned
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write('\n')
    
    print(f"Created {OUTPUT_PATH} with {len(pruned)} medications")
    
    # Print some stats
    print(f"\nTarget range: 1500-2000")
    print(f"Actual count: {len(pruned)}")
    
    # Show sample including brand names
    print(f"\nSample medications (showing brand names):")
    brand_examples = [item for item in pruned if any(b.casefold() in item['label'].casefold() for b in ['Lexapro', 'Tylenol', 'Advil', 'Aleve', 'Zoloft', 'Prozac', 'Lipitor'])]
    for item in brand_examples[:20]:
        print(f"  {item['label']}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
