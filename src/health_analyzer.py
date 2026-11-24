import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def berakna_sammanfattning(df):
    """Beräknar medelvärde, median, min och max för hälsovariablerna"""
    variabler = ["age", "weight", "height", "systolic_bp", "cholesterol"]

    sammanfattning = df[variabler].agg(
        ["mean", "median", "min", "max"]
    ).T

    sammanfattning = sammanfattning.rename(
        index={
            "age": "Ålder (år)",
            "weight": "Vikt (kg)",
            "height": "Längd (cm)",
            "systolic_bp": "Systoliskt blodtryck (mmHg)",
            "cholesterol": "Kolesterol (mmol/L)",
        },
        columns={
            "mean": "Medelvärde",
            "median": "Median",
            "min": "Min",
            "max": "Max",
        },
    )

    return sammanfattning.round(2)


class HealthAnalyzer:
    """Utför analyser och visualiseringar av hälsostudiens data."""

    def __init__(self, df, bilder_mapp):
        """Sparar datan och mappen där diagram ska sparas"""
        self.df = df
        self.bilder_mapp = Path(bilder_mapp)
        self.bilder_mapp.mkdir(parents=True, exist_ok=True)

    def descriptive_statistics(self):
        """Returnerar en tabell med beskrivande statistik för hälsovariablerna."""
        return berakna_sammanfattning(self.df)
    
    def plot_blood_pressure(self):
        """Ritar ett histogram över systoliskt blodtryck och markerar medelvärdet."""
        blodtryck = self.df["systolic_bp"].dropna()
        andel_i_området = blodtryck.between(135, 167).mean()
        medel = blodtryck.mean()
    
        plt.figure(figsize=(8,5))

        sns.histplot(
            x=blodtryck, 
            bins=20, 
            kde=True,
            color="#A84C7A",
            edgecolor="white",
            linewidth=1
        )

        plt.axvline(
            medel,
            color="#6E043F", 
            linestyle="--", 
            linewidth=2,
            label=f"Medelvärde:{medel:.1f} mmHg"
        )

        plt.axvspan(
            135, 
            167, 
            color="#F2C14E", 
            alpha=0.18,
            label=f"Andel mellan 135 och 167 mmHg: {andel_i_området:.1%}"
        )

        plt.title("Fördelningen av systoliskt blodtryck", fontsize=16, weight="bold")
        plt.xlabel("Systoliskt blodtryck (mmHg)", fontsize=12)
        plt.ylabel("Antal deltagare", fontsize=12)
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        sns.despine()
        plt.tight_layout()

        plt.show()

    def regression_bp_age(self):
        """Räknar ut och visar sambandet mellan ålder och blodtryck"""

        data = self.df[["age", "systolic_bp"]].dropna()

        ålder = data["age"].to_numpy()
        blodtryck = data["systolic_bp"].to_numpy()

        matris = np.column_stack((np.ones(len(ålder)), ålder))
        koefficienter,*_= np.linalg.lstsq(matris, blodtryck, rcond=None)

        startvärde = koefficienter [0]
        lutning = koefficienter[1]

        förutsagt = matris @ koefficienter

        sortering = np.argsort(ålder)

        plt.figure(figsize=(9, 6))
        plt.scatter(
            ålder,
            blodtryck,
            alpha=0.35,
            color="#A84C7A",
            label="Deltagare",
        )
        plt.plot(
            ålder[sortering],
            förutsagt[sortering],
            color="#6E043F",
            linewidth=2.5,
            label="Linjär trend",
        )

        plt.title("Systoliskt blodtryck efter åldern")
        plt.xlabel("Ålder (år)")
        plt.ylabel("Systoliskt blodtryck (mmHg)")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.bilder_mapp / "blodtryck_efter_alder.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
        
        # Returnera modellens resultat så att de kan användas i notebooken 
        return {
            "startvärde": startvärde, 
            "lutning_mmHg_per_år": lutning,
            "förutsagda_blodtryck": förutsagt,
            "ålder": ålder,
            "observerat_blodtryck": blodtryck,
        }

    def multiple_regression_bp(self):
        """Undersöker hur ålder, vikt och kolesterol hänger ihop med blodtryck."""
        
        # Behåll rader där alla variabler som behövs har ett värde.
        data = self.df[
            ["age", "weight", "cholesterol", "systolic_bp"]
        ].dropna()

        # Samla ålder, vikt och kolesterol i Numpy-matris.
        # Det här är faktorerna som modellen använder. 
        faktorer = data[["age", "weight", "cholesterol"]].to_numpy()

        # Lägg till en kolumn med ettor för modellens startvärde.
        matris = np.column_stack((np.ones(len(faktorer)), faktorer))

        #Blodtrycken är värdena som modellen försöker uppskatta. 
        blodtryck = data["systolic_bp"].to_numpy()

        # Räkna ut koefficienterna som passar data bäst. 
        koefficienter, *_= np.linalg.lstsq(matris, blodtryck, rcond=None)

        # Använd koefficienterna för att räkna ut modellens uppskattade blodtryck.
        uppskattat_blodtryck = matris @ koefficienter

        # Mät den total variationen i deltagarnas blodtryck. 
        variation_totalt = np.sum((blodtryck - blodtryck.mean()) ** 2)
        
        # Mät hur mycket variation som finns kvar efter modellens uppskattningar. 
        variation_kvar = np.sum((blodtryck - uppskattat_blodtryck) ** 2)
        
        # R² visar hur stor del av variationen modellen beskriver. 
        r2 = 1 - variation_kvar / variation_totalt

        # Rita faktiskt blodtryck mot modellens uppskattning.
        plt.figure(figsize=(8, 5))
        plt.scatter(
            blodtryck,
            uppskattat_blodtryck,
        alpha=0.5,
        color="#A84C7A",
        label="En punkt per deltagare",
        )

        # Den streckade linjen visar en perfekt uppskattning.
       
        lägsta = min(blodtryck.min(), uppskattat_blodtryck.min())
        högsta = max(blodtryck.max(), uppskattat_blodtryck.max())
        
        plt.plot(
            [lägsta, högsta],
            [lägsta, högsta],
            linestyle="--",
            color="#6E043F",
            label="Uppskattat = faktiskt",
        )

        plt.xlim(lägsta, högsta)
        plt.ylim(lägsta, högsta)

        plt.title("Faktiskt och uppskattat blodtryck")
        plt.xlabel("Faktiskt blodtryck (mmHg)")
        plt.ylabel("Uppskattat blodtryck (mmHg)")
        plt.text(
            lägsta + (högsta - lägsta) * 0.05,
            högsta - (högsta - lägsta) * 0.08,
            f"R² = {r2:.3f}",
        )
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
        self.bilder_mapp / "multipel_regression_blodtryck.png",
        dpi=150,
        bbox_inches="tight",
        )

        plt.show()

        # Residualen är skillnaden mellan faktiskt och uppskattat blodtryck.
        residualer = blodtryck - uppskattat_blodtryck

        # Rita modellens fel vid olika uppskattade blodtryck.
        plt.figure(figsize=(8, 5))
        plt.scatter(
        uppskattat_blodtryck,
        residualer,
        alpha=0.5,
        color="#A84C7A",
        )
        plt.axhline(0, color="#6E043F", linestyle="--")

        plt.title("Modellens fel vid olika uppskattningar")
        plt.xlabel("Uppskattat blodtryck (mmHg)")
        plt.ylabel("Faktiskt minus uppskattat (mmHg)")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
        self.bilder_mapp / "residualdiagram_blodtryck.png",
        dpi=150,
        bbox_inches="tight",
        )
        plt.show()

        # Skicka tillbaka startvärdet, koefficienterna och R².
        return {
            "startvärde": koefficienter[0],
            "ålder": koefficienter[1],
            "vikt":koefficienter[2],
            "kolesterol": koefficienter[3],
            "r2": r2, 
        }