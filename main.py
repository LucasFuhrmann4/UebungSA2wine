import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# 1. Daten einlesen

file_path = "winequality-red.csv"

df = pd.read_csv(file_path, delimiter=';')

# 2. Datensatz erkunden

print("\nErste 5 Zeilen des DataFrames:")

print(df.head())  # Erste Zeilen anzeigen

print("\nLetzte 5 Zeilen des DataFrames:")

print(df.tail())  # Letzte Zeilen anzeigen

# 3. Zusammenfassung des DataFrames

print("\nAllgemeine Informationen zum DataFrame:")

print(df.info())

print("\nStatistische Übersicht des DataFrames:")

print(df.describe())

# 4. Spalten auswählen

selected_columns = df[['alcohol', 'pH']]

print("\nErste 10 Zeilen der ausgewählten Spalten (Alkohol & pH):")

print(selected_columns.head(10))

# 5. Bedingte Auswahl (quality genau 8)

filtered_df_8 = df[df['quality'] == 8]

print("\nZeilen mit Qualität genau 8:")

print(filtered_df_8)

# 6. Mehrere Bedingungen (Alkoholgehalt > 12.5 und Qualität >= 7)

filtered_df = df[(df['alcohol'] > 12.5) & (df['quality'] >= 7)]

print("\nZeilen mit Alkoholgehalt > 12.5 und Qualität >= 7:")

print(filtered_df)

# 7. Neue Spalte hinzufügen (Dichte / Alkoholgehalt)

df['density_alcohol_ratio'] = df['density'] / df['alcohol']

print("\nErste Zeilen mit neuer Spalte (Dichte/Alkoholgehalt):")

print(df.head())


# 8. Werte in der quality-Spalte ändern


def quality_label(q):
    if q == 3:

        return "sehr schlecht"

    elif q == 4:

        return "schlecht"

    elif q == 5:

        return "okay"

    elif q == 6:

        return "gut"

    else:

        return "sehr gut"


df['quality_label'] = df['quality'].apply(quality_label)

print("\nZuordnung der Qualitätswerte:")

print(df[['quality', 'quality_label']].head())

# 9. Entfernen der Zeilen mit pH-Wert kleiner als 3.0

df = df[df['pH'] >= 3.0]

print("\nErste Zeilen nach Entfernen von pH-Werten < 3.0:")

print(df.head())

# 10. Spaltenüberschriften ausgeben

print("\nSpaltenüberschriften des DataFrames:")

print(df.columns)

# 11. Spaltenüberschriften mit deutschen Bezeichnungen ersetzen

deutsch_columns = {

    'fixed acidity': 'fester Säuregehalt',

    'volatile acidity': 'flüchtiger Säuregehalt',

    'citric acid': 'Zitronensäure',

    'residual sugar': 'Restzucker',

    'chlorides': 'Chloride',

    'free sulfur dioxide': 'freies Schwefeldioxid',

    'total sulfur dioxide': 'Gesamtschwefeldioxid',

    'density': 'Dichte',

    'pH': 'pH-Wert',

    'sulphates': 'Sulfate',

    'alcohol': 'Alkohol',

    'quality': 'Qualität'

}

df.rename(columns=deutsch_columns, inplace=True)

print("\nErste Zeilen nach Umbenennung der Spalten:")

print(df.head())

# 12. Visualisierung (Scatterplot von Alkohol vs. Qualität)

print("\nErstelle Scatterplot für Alkohol vs. Qualität...")

sns.scatterplot(x=df['Alkohol'], y=df['Qualität'])

plt.xlabel("Alkoholgehalt")

plt.ylabel("Qualität")

plt.title("Alkohol vs. Qualität")

plt.show()

# 13. KMeans Qualität

print("\n#13 KMeans Qualität")

# Daten ohne Zielspalte erzeugen

data_unknown = df.drop(['Qualität', 'quality_label'], axis=1)

print(data_unknown.dtypes)

model = KMeans()

# Ermittle optimale Anzahl von Klassen

visualizer = KElbowVisualizer(model, k=(2, 9))

visualizer.fit(data_unknown)

visualizer.show()

# n_clusters wird nach der graphischen Analyse initialisiert

kmeans = KMeans(n_clusters=4)

# Vorhersage der Klassen

pred = kmeans.fit_predict(data_unknown)

# Zusammenfügen der Datensätze (Spalten)

data_new = pd.concat([df, pd.DataFrame(pred, columns=['label'])], axis=1)

print(data_new)

# Speichern in neue CSV Datei

data_new.to_csv("./data_new.csv")
