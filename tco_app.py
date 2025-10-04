"""
Streamlit — Calculateur TCO BEV vs ICE (MVP ++)

Dépendances (requirements.txt):
streamlit>=1.37
pandas>=2.2
numpy>=1.26
plotly>=5.22

Lancement local:
pip install -r requirements.txt
streamlit run app_streamlit_tco.py

Déploiement (Streamlit Community Cloud):
- Pousse ce fichier + requirements.txt sur un repo GitHub
- Crée l’app sur streamlit.io > Deploy > choisis ton repo/branche/fichier
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st




# ------------------- UI Setup ---------------------
st.set_page_config(
    page_title="Calculateur TCO - BEV vs ICE",
    page_icon="⚡",
    layout="wide",
)

st.title("Calculateur TCO - BEV vs ICE")
st.caption(
    "Comparer le coût total de possession (TCO) d'un véhicule électrique (BEV) vs thermique (ICE).\n"
    "Version MVP : hypothèses paramétrables, actualisation et inflation basiques, dataviz interactive."
)

# ------------------- Paramètres communs ---------------------
st.sidebar.header("Paramètres communs")

years = st.sidebar.slider("Horizon (années)", min_value=3, max_value=15, value=8)

km_mode = st.sidebar.radio("Kilométrage", ["Constant", "Personnalisé"], horizontal=True)

if km_mode == "Constant":
    km_year = st.sidebar.number_input("km / an (constant)", min_value=0, max_value=100_000, value=20_000, step=1_000)
    km_vector = [km_year] * years
else:
    st.sidebar.caption("Saisis le kilométrage annuel par année :")
    km_vector = []
    for i in range(years):
        km_vector.append(
            st.sidebar.number_input(f"Année {i + 1}", min_value=0, max_value=100_000, value=10_000, step=1_000, key=f"km_{i}")
        )

# ------------------- Paramètres économiques ---------------------
st.sidebar.header("Paramètres économiques")

discount_rate = st.sidebar.number_input(
    "Taux d'actualisation (%)",
    min_value=0.0, max_value=15.0, value=4.0, step=0.5
) / 100

energy_inflation = st.sidebar.number_input(
    "Inflation énergie (%)",
    min_value=-5.0, max_value=50.0, value=2.0, step=0.5
) / 100

opex_inflation = st.sidebar.number_input(
    "Inflation maintenance & assurance (%)",
    min_value=-5.0, max_value=20.0, value=1.5, step=0.5
) / 100


# ------------------- Dataclass véhicule ---------------------
@dataclass
class Vehicle:
    name: str
    purchase_price: float
    consumption_per_100km: float
    energy_price_start: float
    maintenance_per_km_start: float
    insurance_per_year_start: float
    residual_rate: float


# ------------------- Entrées véhicules ---------------------
st.subheader("Hypothèses véhicules")
col_bev, col_ice = st.columns(2)

with col_bev:
    st.markdown("**BEV - Véhicule électrique**")
    bev = Vehicle(
        name="BEV",
        purchase_price=st.number_input("Prix d'achat (CHF)", 0, 200_000, 55_000, step=1_000),
        consumption_per_100km=st.number_input("Conso (kWh/100 km)", 0.0, 40.0, 17.0, step=0.5),
        energy_price_start=st.number_input("Prix de l'électricité initial (cts/kWh)", 0.0, 500.0, 45.0, step=1.0),
        maintenance_per_km_start=st.number_input("Maintenance (CHF/km)", 0.0, 1.0, 0.03, step=0.01),
        insurance_per_year_start=st.number_input("Assurance (CHF/an)", 0.0, 10_000.0, 900.0, step=50.0),
        residual_rate=st.number_input("Taux valeur résiduelle (%)", 0.0, 80.0, 35.0, step=5.0) / 100,
    )

with col_ice:
    st.markdown("**ICE - Véhicule thermique**")
    ice = Vehicle(
        name="ICE",
        purchase_price=st.number_input("Prix d'achat (CHF)", 0, 200_000, 42_000, step=1_000),
        consumption_per_100km=st.number_input("Conso (L/100 km)", 0.0, 25.0, 10.0, step=0.5),
        energy_price_start=st.number_input("Prix du carburant initial (CHF/L)", 0.0, 5.0, 1.95, step=0.01),
        maintenance_per_km_start=st.number_input("Maintenance (CHF/km)", 0.0, 2.0, 0.06, step=0.01),
        insurance_per_year_start=st.number_input("Assurance (CHF/an)", 0.0, 10_000.0, 1_000.0, step=50.0),
        residual_rate=st.number_input("Taux valeur résiduelle (%)", 0.0, 80.0, 25.0, step=5.0) / 100,
    )


# ------------------- Utilitaires de calcul ---------------------
def grow_series(start_value: float, growth_rate: float, n_years: int) -> list[float]:
    """
    Liste annuelle avec croissance composée.
    Exemple : grow_series(1.0, 0.1, 3) -> [1.0, 1.1, 1.21]
    """
    return [start_value * ((1 + growth_rate) ** t) for t in range(n_years)]


def annual_energy_cost(consumption_per_100km: float, unit_price: float, km: float, vehicle_type: str) -> float:
    """
    Coût annuel de l'énergie = (conso/100) * km * prix_unitaire
    - BEV : unit_price saisi en cts/kWh -> converti en CHF/kWh
    - ICE : unit_price saisi en CHF/L
    """
    if vehicle_type == "BEV":
        price_chf = unit_price / 100.0  # cts -> CHF
    else:
        price_chf = unit_price
    return (consumption_per_100km / 100.0) * km * price_chf


def npv(cashflows: list[float], r: float) -> float:
    """
    Valeur actualisée nette des flux.
    cashflows[0] = flux année 0 (non actualisé)
    """
    if r == 0:
        return float(sum(cashflows))
    return float(sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows)))


# ------------------- Noyau de calcul TCO ---------------------
def compute_tco(
    v: Vehicle,
    years: int,
    km_per_year: list[float] | float,
    discount_rate: float,
    energy_inflation: float,
    opex_inflation: float,
) -> dict:
    """
    Calcule les flux financiers du TCO d'un véhicule.
    Retourne un dict avec :
      - npv_total_cost
      - tco_per_km
      - annual_table (DataFrame)
      - residual_value (brute, année finale)
      - total_km
    """
    # 1) Normaliser km
    if isinstance(km_per_year, (int, float)):
        km_vec = [float(km_per_year)] * years
    else:
        km_vec = list(km_per_year)
    assert len(km_vec) == years, "La liste km_per_year doit avoir une valeur par année"

    # 2) Séries prix/opex
    energy_price_series = grow_series(v.energy_price_start, energy_inflation, years)
    maint_per_km_series = grow_series(v.maintenance_per_km_start, opex_inflation, years)
    insurance_series = grow_series(v.insurance_per_year_start, opex_inflation, years)

    # 3) Flux année 0 (achat)
    cashflows = [-v.purchase_price]
    rows = []

    # 4) Flux annuels OPEX
    for t in range(1, years + 1):
        km = km_vec[t - 1]
        energy_cost = annual_energy_cost(v.consumption_per_100km, energy_price_series[t - 1], km, v.name)
        maint_cost = maint_per_km_series[t - 1] * km
        insurance_cost = insurance_series[t - 1]
        opex_total = energy_cost + maint_cost + insurance_cost

        cf = -opex_total  # coût
        rows.append({
            "Année": t,
            "km": km,
            "Prix énergie": energy_price_series[t - 1],
            "Coûts énergie": energy_cost,
            "Maintenance": maint_cost,
            "Assurance": insurance_cost,
            "OPEX": opex_total,   # positif (coût)
            "Cashflow": cf,       # négatif
        })
        cashflows.append(cf)

    # 5) Valeur résiduelle (brute, ajoutée à la dernière année)
    residual_value = v.purchase_price * v.residual_rate
    cashflows[-1] += residual_value                      # flux net année finale
    rows[-1]["Valeur résiduelle"] = residual_value       # info brute (non actualisée)
    rows[-1]["Cashflow"] += residual_value               # reflète le flux net année finale

    # 6) NPV & TCO
    total_km = sum(km_vec)
    total_npv = npv(cashflows, discount_rate)            # négatif typiquement (coût)
    tco_per_km = -total_npv / total_km if total_km > 0 else math.inf

    # 7) Table annuelle + cumul NPV
    df = pd.DataFrame(rows)
    # Actualisation des flux (hors année 0)
    df["Cashflow actualisé"] = [cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cashflows[1:], start=1)]
    df["Cumul NPV"] = df["Cashflow actualisé"].cumsum() + cashflows[0]

    return {
        "vehicle": v,
        "cashflows": cashflows,
        "npv_total_cost": total_npv,
        "tco_per_km": tco_per_km,
        "annual_table": df,
        "residual_value": residual_value,  # brute
        "total_km": total_km,
    }


# ------------------- Calculs ---------------------
bev_res = compute_tco(bev, years, km_vector, discount_rate, energy_inflation, opex_inflation)
ice_res = compute_tco(ice, years, km_vector, discount_rate, energy_inflation, opex_inflation)

st.subheader("Résultats comparés")
diff_npv = bev_res["npv_total_cost"] - ice_res["npv_total_cost"]
diff_tco = bev_res["tco_per_km"] - ice_res["tco_per_km"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("TCO BEV (CHF/km)", f"{bev_res['tco_per_km']:.2f}")
col2.metric("TCO ICE (CHF/km)", f"{ice_res['tco_per_km']:.2f}")
col3.metric("Δ TCO (BEV − ICE)", f"{diff_tco:.2f} CHF/km")
col4.metric("Δ NPV total (BEV − ICE)", f"{diff_npv:,.0f} CHF")

st.write("BEV - NPV (coût actualisé):", f"{abs(bev_res['npv_total_cost']):,.0f} CHF")
st.write("ICE - NPV (coût actualisé):", f"{abs(ice_res['npv_total_cost']):,.0f} CHF")


# ------------------- Dataviz ---------------------
st.subheader("Visualisation des coûts")

# --- Décomposition TCO (CAPEX net + OPEX actualisés) ---
def decompose_tco(res: dict, label: str, discount_rate: float, years: int) -> pd.DataFrame:
    df = res["annual_table"].copy()

    # OPEX actualisés : OPEX(t) / (1+r)^t (reste positif ; c'est un coût)
    df["OPEX actualisé"] = df["OPEX"] / ((1 + discount_rate) ** df["Année"])
    opex_actualises = float(df["OPEX actualisé"].sum())

    # Valeur résiduelle actualisée (recette, positive)
    residual_disc = res["residual_value"] / ((1 + discount_rate) ** years)

    # CAPEX net (positif pour l'affichage) = Achat - Valeur résiduelle actualisée
    capex_net = res["vehicle"].purchase_price - residual_disc

    # Test de cohérence : |NPV| doit égaler CAPEX_net + OPEX_actualisés
    npv_abs = abs(res["npv_total_cost"])
    tco_bar_sum = capex_net + opex_actualises
    if abs(npv_abs - tco_bar_sum) > 1e-6:
        st.caption(
            f"⚠️ Alerte cohérence {label} : |NPV|={npv_abs:,.0f} vs CAPEX_net+OPEX={tco_bar_sum:,.0f} "
            f"(écart {npv_abs - tco_bar_sum:,.2f})"
        )

    return pd.DataFrame({
        "Véhicule": [label] * 2,
        "Poste": ["CAPEX net (achat - revente actualisée)", "OPEX actualisés"],
        "Coût actualisé": [capex_net, opex_actualises],
    })


df_tco = pd.concat([
    decompose_tco(bev_res, "BEV", discount_rate, years),
    decompose_tco(ice_res, "ICE", discount_rate, years),
], ignore_index=True)

# Ajouter des descriptions pour les tooltips
tooltip_map = {
    "CAPEX net (achat - revente actualisée)": "Coûts d’investissement, achat du véhicule moins valeur de revente actualisée.",
    "OPEX actualisés": "Coûts d’exploitation : énergie, maintenance et assurance, actualisés sur la période."
}

df_tco["Description"] = df_tco["Poste"].map(tooltip_map)

totaux_tco = df_tco.groupby("Véhicule", as_index=False)["Coût actualisé"].sum()

fig_total = px.bar(
    df_tco,
    x="Véhicule",
    y="Coût actualisé",
    color="Poste",
    barmode="stack",
    title="Décomposition du TCO actualisé (BEV vs ICE) — CAPEX net + OPEX",
    text_auto=".0f",
    hover_data={
        "Poste":False,
        "Coût actualisé": False,
        "Description": True,
        "Véhicule": False
    }
)

for _, row in totaux_tco.iterrows():
    fig_total.add_annotation(
        x=row["Véhicule"],
        y=row["Coût actualisé"],
        text=f"{row['Coût actualisé']:,.0f} CHF",
        showarrow=False,
        font=dict(size=14, color="black"),  # <<< couleur explicite
        yshift=10,
    )

fig_total.update_layout(
    plot_bgcolor="white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, title="CHF (actualisés)"),
    title=dict(x=0, xanchor="left", font=dict(size=20)),
    bargap=0.3,
)

# Couleur de texte selon thème Streamlit
_is_dark = (st.get_option("theme.base") == "dark")
FONT_COLOR = "white" if _is_dark else "black"

# Construire un customdata pour un tooltip propre
import numpy as np
customdata = np.stack([df_tco["Poste"], df_tco["Description"]], axis=-1)

fig_total.update_traces(
    customdata=customdata,
    hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>%{customdata[1]}<br>"
                  "Total: <b>CHF %{y:,.0f}</b><extra></extra>"
)

fig_total.update_layout(
    paper_bgcolor="rgba(255, 255, 255, 1)",   # fond transparent
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=FONT_COLOR),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=FONT_COLOR)),
    yaxis=dict(showgrid=False, zeroline=False, title="CHF (actualisés)", tickfont=dict(color=FONT_COLOR)),
    title=dict(x=0, xanchor="left", font=dict(size=20, color=FONT_COLOR)),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_COLOR),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1.0
    ),
    bargap=0.3,
)

# Rendre le label total plus lisible (cartouche blanc translucide)
for _, row in totaux_tco.iterrows():
    fig_total.add_annotation(
        x=row["Véhicule"], y=row["Coût actualisé"],
        text=f"{row['Coût actualisé']:,.0f} CHF",
        showarrow=False,
        font=dict(size=14, color=FONT_COLOR),
        yshift=10,
        bgcolor="rgba(255,255,255,0.6)" if _is_dark else "rgba(0,0,0,0.05)",
        bordercolor="rgba(0,0,0,0)",
        borderpad=2,
    )



st.plotly_chart(fig_total, use_container_width=True)


# --- Courbe cumul des coûts actualisés (positif) ---
# Construire cum_df
cum_b = bev_res["annual_table"][["Année", "Cumul NPV"]].copy()
cum_b["Véhicule"] = "BEV"
cum_i = ice_res["annual_table"][["Année", "Cumul NPV"]].copy()
cum_i["Véhicule"] = "ICE"
cum_df = pd.concat([cum_b, cum_i], ignore_index=True)

# Coûts en positif pour la lecture (cumul des coûts actualisés)
cum_df["Cumul NPV positif"] = cum_df["Cumul NPV"].abs()

fig_line = px.line(
    cum_df,
    x="Année",
    y="Cumul NPV positif",
    color="Véhicule",
    title="Cumul des coûts actualisés",
    markers=True,
)

fig_line.update_traces(line=dict(width=3), marker=dict(size=8))

fig_line.update_layout(
    paper_bgcolor="rgb(255, 255, 255)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=FONT_COLOR),
    title=dict(x=0, xanchor="left", font=dict(size=15, color=FONT_COLOR)),
    xaxis=dict(
        title="Année",
        showgrid=False, zeroline=False,
        tickfont=dict(color=FONT_COLOR),
    ),
    yaxis=dict(
        title="CHF (cumulé)",
        showgrid=False, zeroline=False,
        rangemode="tozero",  # démarre à 0
        tickfont=dict(color=FONT_COLOR),
    ),
    legend=dict(
        orientation="h",
        y=-0.2, x=0.5, xanchor="center",
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_COLOR),
    ),
)

st.plotly_chart(fig_line, use_container_width=False)

st.write(df_tco)



