# Tennis Injury Risk Prediction

## Opis projektu

Celem projektu jest analiza danych biomechanicznych związanych z grą w tenisa oraz zbudowanie modelu przewidującego poziom ryzyka kontuzji.

W projekcie sprawdzam, które cechy związane z chwytem rakiety, siłą uderzenia, wibracjami, zmęczeniem mięśni i stabilnością chwytu mogą mieć znaczenie przy przewidywaniu ryzyka kontuzji.

Zmienną przewidywaną jest `Injury_Risk_Level`, czyli poziom ryzyka kontuzji.

## Zbiór danych

Dane pochodzą ze zbioru **NanoGrip Tennis Biomechanics Dataset** dostępnego na Kaggle.

Zbiór zawiera informacje o pomiarach biomechanicznych wykonywanych podczas uderzeń tenisowych.

Początkowo zbiór danych zawierał:

- 4780 obserwacji,
- 28 kolumn,
- dane numeryczne i kategoryczne.

Przykładowe cechy w danych:

- typ uderzenia,
- typ nawierzchni,
- siła chwytu,
- stabilność chwytu,
- zmęczenie mięśni,
- siła uderzenia,
- wibracje,
- poślizg chwytu,
- poziom ryzyka kontuzji.

## Cel analizy

Celem projektu było:

- wczytanie i sprawdzenie danych,
- oczyszczenie danych,
- analiza rozkładu poziomów ryzyka kontuzji,
- sprawdzenie zależności między cechami biomechanicznymi a ryzykiem kontuzji,
- zbudowanie modeli klasyfikacyjnych,
- porównanie wyników modeli,
- sprawdzenie najważniejszych cech dla modelu.

## Użyte technologie

W projekcie wykorzystałam:

- Python,
- pandas,
- numpy,
- matplotlib,
- seaborn,
- scikit-learn,
- joblib.

## Przygotowanie danych

W pierwszym kroku dane zostały wczytane i sprawdzone pod kątem brakujących wartości oraz duplikatów.

Z danych zostały usunięte kolumny identyfikacyjne:

- `Subject_ID`,
- `Trial_ID`.

Nie były one potrzebne do analizy ani modelowania, ponieważ służyły tylko do identyfikacji osoby i próby.

Po czyszczeniu danych zbiór zawierał:

- 4780 obserwacji,
- 26 kolumn.

## Eksploracyjna analiza danych

W analizie sprawdziłam m.in.:

- rozkład poziomów ryzyka kontuzji,
- stabilność chwytu w zależności od poziomu ryzyka,
- zmęczenie mięśni w zależności od poziomu ryzyka,
- siłę uderzenia w zależności od poziomu ryzyka,
- występowanie poślizgu chwytu,
- zależności między cechami biomechanicznymi.

### Rozkład poziomu ryzyka kontuzji

![Rozkład poziomu ryzyka kontuzji](images/01_injury_risk_distribution.png)

Rozkład zmiennej `Injury_Risk_Level`:

| Poziom ryzyka | Liczba obserwacji |
|---|---:|
| High | 2027 |
| Low | 1432 |
| Medium | 1321 |

W danych najczęściej występował wysoki poziom ryzyka kontuzji.

### Zmęczenie mięśni a ryzyko kontuzji

![Zmęczenie mięśni a poziom ryzyka kontuzji](images/03_muscle_fatigue_by_risk.png)

Indeks zmęczenia mięśni opisuje poziom zmęczenia mięśni podczas ruchu. W projekcie jest to jedna z cech, które mogą pomagać w odróżnianiu poziomów ryzyka kontuzji.

### Poślizg chwytu a ryzyko kontuzji

![Poślizg chwytu a poziom ryzyka kontuzji](images/05_slip_events_by_risk.png)

Poślizg chwytu informuje, czy podczas próby wystąpiło przesunięcie lub utrata stabilności chwytu rakiety.

## Modelowanie

Do przewidywania poziomu ryzyka kontuzji wykorzystałam dwa modele klasyfikacyjne:

- regresję logistyczną,
- las losowy.

Zmienną przewidywaną była kolumna:

- `Injury_Risk_Level`

Modele przewidywały jedną z trzech klas:

- `Low`,
- `Medium`,
- `High`.

Dane zostały podzielone na zbiór treningowy i testowy.

## Wyniki modeli

| Model | Dokładność |
|---|---:|
| Regresja logistyczna | 0.7134 |
| Las losowy | 0.9969 |

Najlepszy wynik uzyskał model **lasu losowego**.

### Porównanie dokładności modeli

![Porównanie dokładności modeli](images/10_model_comparison.png)

Las losowy osiągnął bardzo wysoką dokładność. Oznacza to, że w tym zbiorze danych model bardzo dobrze rozpoznawał poziom ryzyka kontuzji na podstawie dostępnych cech.

## Najważniejsze cechy modelu

Dla modelu lasu losowego sprawdziłam ważność cech.

Najważniejsze cechy to:

- szczyt wibracji,
- zmienność siły chwytu,
- indeks zmęczenia mięśni,
- pole kontaktu,
- temperatura.

![Najważniejsze cechy modelu](images/09_feature_importance.png)

Wyniki sugerują, że model najczęściej korzystał z informacji związanych z wibracjami, siłą chwytu i zmęczeniem mięśni.

## Wnioski

Na podstawie analizy można zauważyć, że:

- cechy biomechaniczne mogą być przydatne w przewidywaniu ryzyka kontuzji,
- szczególnie ważne okazały się cechy związane z wibracjami, siłą chwytu i zmęczeniem mięśni,
- las losowy uzyskał dużo lepszy wynik niż regresja logistyczna,
- model lasu losowego bardzo dobrze dopasował się do danych testowych.

Model nie uwzględnia wielu czynników, które mogłyby mieć znaczenie w prawdziwym sporcie, np. historii kontuzji, techniki gry, obciążenia treningowego, regeneracji czy poziomu zmęczenia w dłuższym czasie.


