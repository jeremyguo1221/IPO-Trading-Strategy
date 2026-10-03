import pandas as pd

OUT = r"C:\Users\Jeremy Guo\Desktop\Current Coding Project\ipo_first_6_months"
DB_END = pd.Timestamp("2026-07-31")

# symbol: (company, sector, offer price USD)
META = {
 "WRBY":("Warby Parker","Consumer",None),"GTLB":("GitLab","Technology",77),"PTLO":("Portillo's","Consumer",20),
 "RENT":("Rent the Runway","Consumer",21),"GFS":("GlobalFoundries","Semiconductors & AI hardware",47),
 "FLNC":("Fluence Energy","Energy & Power",28),"BIRD":("Allbirds","Consumer",15),"NRDS":("NerdWallet","Financials & Fintech",18),
 "ARHS":("Arhaus","Consumer",12),"RIVN":("Rivian","Autos & Mobility",78),"BRZE":("Braze","Technology",65),
 "SG":("Sweetgreen","Consumer",28),"NU":("Nu Holdings","Financials & Fintech",9),"IOT":("Samsara","Technology",23),
 "TPG":("TPG","Financials & Fintech",29),"CRDO":("Credo Technology","Semiconductors & AI hardware",10),
 "BLCO":("Bausch + Lomb","Healthcare",18),"CRBG":("Corebridge Financial","Financials & Fintech",21),
 "MBLY":("Mobileye","Autos & Mobility",21),"NXT":("Nextracker","Energy & Power",24),"KVUE":("Kenvue","Consumer",22),
 "ATMU":("Atmus Filtration","Industrials & Aerospace",19.5),"CAVA":("Cava","Consumer",22),"SVV":("Savers Value Village","Consumer",18),
 "KGS":("Kodiak Gas Services","Energy & Power",16),"ODD":("Oddity Tech","Consumer",35),"ARM":("Arm Holdings","Semiconductors & AI hardware",51),
 "CART":("Instacart","Technology",30),"KVYO":("Klaviyo","Technology",30),"BIRK":("Birkenstock","Consumer",46),
 "AS":("Amer Sports","Consumer",13),"ALAB":("Astera Labs","Semiconductors & AI hardware",36),"RDDT":("Reddit","Technology",34),
 "ULS":("UL Solutions","Industrials & Aerospace",28),"IBTA":("Ibotta","Technology",88),"RBRK":("Rubrik","Technology",32),
 "LOAR":("Loar Holdings","Industrials & Aerospace",28),"VIK":("Viking Holdings","Consumer",24),"WAY":("Waystar","Healthcare",21),
 "TEM":("Tempus AI","Healthcare",37),"WBTN":("Webtoon Entertainment","Technology",21),"LINE":("Lineage","Real Estate & Infra",78),
 "SARO":("StandardAero","Industrials & Aerospace",24),"KLC":("KinderCare","Consumer",24),"INGM":("Ingram Micro","Technology",22),
 "WRD":("WeRide","Autos & Mobility",15.5),"PONY":("Pony AI","Autos & Mobility",13),"TTAN":("ServiceTitan","Technology",71),
 "VG":("Venture Global","Energy & Power",25),"SFD":("Smithfield Foods","Consumer",20),"SAIL":("SailPoint","Technology",23),
 "CRWV":("CoreWeave","Technology",40),"ETOR":("eToro","Financials & Fintech",52),"HNGE":("Hinge Health","Healthcare",32),
 "CRCL":("Circle Internet","Financials & Fintech",31),"OMDA":("Omada Health","Healthcare",19),"VOYG":("Voyager Technologies","Industrials & Aerospace",31),
 "CHYM":("Chime","Financials & Fintech",27),"CAI":("Caris Life Sciences","Healthcare",21),"FIG":("Figma","Technology",33),
 "FLY":("Firefly Aerospace","Industrials & Aerospace",45),"BLSH":("Bullish","Financials & Fintech",37),"KLAR":("Klarna","Financials & Fintech",40),
 "FIGR":("Figure Technology","Financials & Fintech",25),"GEMI":("Gemini","Financials & Fintech",28),"STUB":("StubHub","Consumer",23.5),
 "NTSK":("Netskope","Technology",19),"FRMI":("Fermi","Real Estate & Infra",21),"NAVN":("Navan","Technology",25),
 "BETA":("Beta Technologies","Industrials & Aerospace",34),"MDLN":("Medline","Healthcare",29),"BTGO":("BitGo","Financials & Fintech",18),
 "EQPT":("EquipmentShare","Industrials & Aerospace",24.5),"YSS":("York Space Systems","Industrials & Aerospace",34),
 "BOBS":("Bob's Discount Furniture","Consumer",17),"FPS":("Forgent Power Solutions","Industrials & Aerospace",27),
 "MWH":("SOLV Energy","Energy & Power",25),"MMED":("MiniMed","Healthcare",20),"PAYP":("PayPay","Financials & Fintech",16),
 "MAIR":("Madison Air Solutions","Industrials & Aerospace",27),"ARXS":("Arxis","Industrials & Aerospace",28),"XE":("X-Energy","Energy & Power",23),
 "PS":("Pershing Square Inc","Financials & Fintech",50),"PSUS":("Pershing Square USA (fund)","Financials & Fintech",50),
 "HAWK":("HawkEye 360","Industrials & Aerospace",26),"FRVO":("Fervo Energy","Energy & Power",27),"BXDC":("Blackstone Digital Infrastructure Trust","Real Estate & Infra",20),
 "CBRS":("Cerebras Systems","Semiconductors & AI hardware",185),"PWRL":("Powerlaw","Financials & Fintech",35),"QNT":("Quantinuum","Technology",60),
 "INIO":("INNIO","Industrials & Aerospace",27),"SPCX":("SpaceX","Industrials & Aerospace",135),"DPC":("Doncasters","Industrials & Aerospace",33),
 "BSP":("Bending Spoons","Technology",29),"LIME":("Lime (Neutron Holdings)","Autos & Mobility",25),"SKHY":("SK hynix (ADR)","Semiconductors & AI hardware",149),
 "CSQR":("Csquare","Real Estate & Infra",21),"JMKE":("Jersey Mike's","Consumer",23),
}
MISSING = {"OLPX":"Olaplex","AVDX":"AvidXchange","INFA":"Informatica","UDMY":"Udemy","HCP":"HashiCorp","ZK":"Zeekr","OS":"OneStream"}

META["PS"] = ("Pershing Square Inc", "Financials & Fintech", None)  # $50 was the PSUS fund price
FLAGS = {
 "RENT": "Prices divided by 20 to undo the 2024 1-for-20 reverse split stored in the DB",
 "BIRD": "Prices divided by 20 to undo the 2024 1-for-20 reverse split stored in the DB",
 "BTGO": "Dropped bogus 2026-01-21 stub bar (419k volume, pre-debut)",
 "ETOR": "Dropped bogus 2025-05-13 stub bar (zero volume at offer price)",
 "KLAR": "DB missing real first trading day (2025-09-10); day-1 figures are day 2",
 "IBTA": "DB missing real first trading day (2024-04-18); day-1 figures are day 2",
 "CRBG": "DB missing real first trading day (2022-09-15); day-1 figures are day 2",
 "INIO": "DB missing first two trading days (2026-06-04/05); day-1 figures are day 3",
 "WRBY": "Direct listing: no offer price",
 "PS": "Offer price not comparable ($50 was the PSUS fund price)",
}

d = pd.read_csv(f"{OUT}/ipo_first_6m_daily.csv", parse_dates=["date"])
rs = d.symbol.isin(["RENT", "BIRD"])
for c in ["open", "high", "low", "close", "adj_close", "vwap"]:
    d.loc[rs, c] = d.loc[rs, c] / 20
d.loc[rs, "volume"] = d.loc[rs, "volume"] * 20
d.insert(1, "company", d.symbol.map(lambda s: META[s][0]))
d.insert(2, "sector", d.symbol.map(lambda s: META[s][1]))
d.to_csv(f"{OUT}/ipo_first_6m_daily.csv", index=False)

def pct(a, b):
    return None if a is None or b is None or pd.isna(a) or pd.isna(b) else round((a / b - 1) * 100, 1)

rows = []
for sym, g in d.groupby("symbol"):
    g = g.sort_values("date")
    name, sector, offer = META[sym]
    c = g.close.to_numpy()
    at = lambda n: c[n - 1] if len(c) >= n else None
    peak = pd.Series(c).cummax()
    dd = ((pd.Series(c) / peak - 1).min()) * 100
    first = g.date.iloc[0]
    complete = first + pd.DateOffset(months=6) <= DB_END + pd.Timedelta(days=1)
    rows.append({
        "symbol": sym, "company": name, "sector": sector, "first_trade_date": first.date(),
        "offer_price": offer, "day1_open": round(g.open.iloc[0], 2), "day1_close": round(c[0], 2),
        "day1_pop_vs_offer_%": pct(c[0], offer),
        "close_day21_(~1m)": None if at(21) is None else round(at(21), 2),
        "close_day63_(~3m)": None if at(63) is None else round(at(63), 2),
        "last_close_in_window": round(c[-1], 2), "last_date_in_window": g.date.iloc[-1].date(),
        "ret_1m_vs_offer_%": pct(at(21), offer), "ret_3m_vs_offer_%": pct(at(63), offer),
        "ret_window_vs_offer_%": pct(c[-1], offer), "ret_window_vs_day1_close_%": pct(c[-1], c[0]),
        "max_close": round(c.max(), 2), "min_close": round(c.min(), 2),
        "max_drawdown_%": round(dd, 1), "trading_days": len(g),
        "full_6_months": "yes" if complete else f"no (data ends {DB_END.date()})",
        "data_note": FLAGS.get(sym, ""),
    })
s = pd.DataFrame(rows).sort_values("first_trade_date")
s.to_csv(f"{OUT}/ipo_first_6m_summary.csv", index=False)
pd.DataFrame([{"symbol": k, "company": v, "reason": "not in ohlcv_daily (acquired/delisted; DB holds current listings only)"} for k, v in MISSING.items()]
             ).to_csv(f"{OUT}/not_in_database.csv", index=False)

full = s[s.full_6_months == "yes"]
print("tickers:", len(s), " full 6m:", len(full))
print("median day1 pop %:", s["day1_pop_vs_offer_%"].median())
print("median 6m vs offer % (full only):", full["ret_window_vs_offer_%"].median(),
      " share above offer:", round((full["ret_window_vs_offer_%"] > 0).mean() * 100))
print(full.groupby("sector")["ret_window_vs_offer_%"].agg(["count", "median"]).round(1).sort_values("median"))
cols = ["symbol", "day1_pop_vs_offer_%", "ret_window_vs_offer_%"]
print("best 6m:\n", full.nlargest(5, "ret_window_vs_offer_%")[cols].to_string(index=False))
print("worst 6m:\n", full.nsmallest(5, "ret_window_vs_offer_%")[cols].to_string(index=False))
print("short-window:", ", ".join(s[s.full_6_months != "yes"].symbol))
# sanity: day1 open vs offer outliers (possible split-adjusted data)
chk = s[s.offer_price.notna()].assign(r=lambda x: x.day1_open / x.offer_price)
print("open/offer ratio outliers:\n", chk[(chk.r < 0.6) | (chk.r > 3)][["symbol", "offer_price", "day1_open", "r"]].to_string(index=False))
