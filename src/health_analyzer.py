import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def berakna_sammanfattning(df):
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

    def __init__(self, df, bilder_mapp):
        self.df = df
        self.bilder_mapp = Path(bilder_mapp)
        self.bilder_mapp.mkdir(parents=True, exist_ok=True)

    def descriptive_statistics(self):
        return berakna_sammanfattning(self.df)

    def plot_blood_pressure(self):
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
        plt.show

        return {
            "startvärde": startvärde, 
            "lutning_mmHg_per_år": lutning,
            "förutsagda_blodtryck": förutsagt,
            "ålder": ålder,
            "observerat_blodtryck": blodtryck,
        }



