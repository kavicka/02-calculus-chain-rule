# Tutoriál · 02 · Derivácie a reťazové pravidlo

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál prejde projekt krok za krokom: čo sa postavilo, prečo je každá časť spravená práve takto a ako funguje. README ukazuje výsledky; tento súbor vysvetľuje cestu k nim.

**Cieľ:** vypočítať gradienty malého výpočtu troma nezávislými spôsobmi (ručne, numericky, cez PyTorch) a ukázať, že sa zhodujú. Je to presne tá matematika, ktorú v každej neurónovej sieti počíta backpropagation, a základ pre vlastný autograd v projekte 04.

**Potrebuješ:** setup z projektu 01 (`uv`, PyTorch). Stredoškolskú algebru. Z analýzy stačí „sklon priamky“.

---

## Krok 1 · Vyber najmenší graf, ktorý má všetko

```python
INPUTS = {"a": 2.0, "b": -3.0, "c": 10.0}

def forward(a, b, c):
    e = a * b      # -6
    d = e + c      #  4
    L = d ** 2     # 16
    return {"e": e, "d": d, "L": L}
```

**Prečo práve tieto tri operácie:** vrstva neurónovej siete je „vynásob váhami, pripočítaj bias, aplikuj nelinearitu, spočítaj loss“. Tento graf má od každého druhu jednu: `×` (váhy), `+` (bias), `²` (loss, ako kvadratická chyba). Keď zvládneš backprop cez tieto tri, zvládneš ho cez sieť.

**Prečo pomenovať každý medzivýsledok (`e`, `d`):** backprop potrebuje medzihodnoty. Keď ich zapíšeš ako pomenované uzly, zo vzorca sa stane **výpočtový graf**: reťaz jednoduchých krokov, každý so vstupom a výstupom. Presne takýto obraz si interne stavia PyTorch.

**Prečo tieto čísla:** `b` je záporné, aby sme videli záporný gradient, a hodnoty sú malé celé čísla, takže každý výsledok overíš z hlavy.

---

## Krok 2 · Pochop, čo je gradient

`∂L/∂a` sa pýta: *keď `a` trošku posuniem, o koľko sa pohne `L` na jednotku posunu?* Je to sklon `L` ako funkcie `a` v aktuálnom bode.

- Sklon −24 znamená: zvýš `a` o 0,001 a `L` klesne zhruba o 0,024.
- **Gradient descent** to využíva priamo: aby si `L` zmenšil, posuň `a` *opačným* smerom, než ukazuje gradient.

[`plot.py`](plot.py) to kreslí ako dotyčnicu ku krivke `L(a)` v bode `a = 2` (`assets/tangent.png`).

---

## Krok 3 · Backprop ručne cez reťazové pravidlo

**Reťazové pravidlo:** ak `L` závisí od `d` a `d` závisí od `e`, potom

```
∂L/∂e = ∂L/∂d · ∂d/∂e
```

Slovami: vplyv `e` na `L` = (vplyv `d` na `L`) × (vplyv `e` na `d`). Každý činiteľ je **lokálna** derivácia: pozerá sa len na jednu operáciu.

Začneme teda na konci, kde `∂L/∂L = 1`, a ideme dozadu a násobíme:

```python
def manual_grads(a, b, c):
    v = forward(a, b, c)
    dL_dL = 1.0
    dL_dd = dL_dL * 2 * v["d"]  # L = d²     → lokálne 2d  → 8
    dL_de = dL_dd * 1.0         # d = e + c  → lokálne 1   → 8
    dL_dc = dL_dd * 1.0         #            → lokálne 1   → 8
    dL_da = dL_de * b           # e = a * b  → lokálne b   → -24
    dL_db = dL_de * a           #            → lokálne a   → 16
```

**Prečo dozadu a nie dopredu:** chceme gradient *jedného* výstupu podľa *mnohých* vstupov. Keď ideme dozadu, gradient každého uzla sa spočíta raz a použije ho všetko, čo do neho vteká. Sieť má milióny vstupov (váh) a jeden loss, takže dozadu je to miliónkrát lacnejšie. V tom je celý trik backpropagation.

**Dve pravidlá, ktoré sa oplatí zapamätať:**
- **`+` skopíruje gradient** do všetkých vstupov bez zmeny (lokálna derivácia je 1). Preto sú `∂L/∂e` aj `∂L/∂c` rovné 8.
- **`×` vymení vstupy**: gradient, ktorý ide do `a`, sa násobí `b`, a do `b` sa násobí `a`. Preto `∂L/∂a = 8 · (−3)`.

Všimni si, že hodnoty z forward passu (`d = 4`, `a`, `b`) sú potrebné počas backward passu. Preto frameworky držia aktivácie v pamäti počas tréningu.

---

## Krok 4 · Over to numericky (bez analýzy)

Úplne nezávislý spôsob, ako dostať sklon: posuň vstup a zmeraj.

```python
def numerical_grad(name, h=1e-6, central=True):
    up = {**INPUTS, name: INPUTS[name] + h}
    if central:
        down = {**INPUTS, name: INPUTS[name] - h}
        return (loss(**up) - loss(**down)) / (2 * h)
    return (loss(**up) - loss(**INPUTS)) / h
```

**Prečo je to užitočné:** nezdieľa s Krokom 3 žiadny kód ani úvahu. Ak sa obe zhodnú, ručné odvodenie je takmer určite správne. Takýto „gradient check“ je spôsob, ako testovať vlastný backward pass (projekt 04 používa rovnakú myšlienku v testoch).

**Prečo centrálna diferencia `(f(x+h) − f(x−h)) / 2h`:** jednostranná verzia `(f(x+h) − f(x)) / h` má chybu úmernú `h`, lebo ignoruje zakrivenie krivky. Symetrická verzia tento člen zakrivenia vyruší, takže jej chyba je úmerná `h²`, čo je oveľa menej.

**Prečo sa nepoužíva na tréning:** stojí jeden forward pass navyše *na každý vstup*. Pri modeli s miliónom váh je to milión forward passov na krok. Backprop ich dostane všetky v jednom backward passe.

---

## Krok 5 · Nechaj to na PyTorch

```python
t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in INPUTS.items()}
e = t["a"] * t["b"]
d = e + t["c"]
L = d ** 2
e.retain_grad(); d.retain_grad()
L.backward()
```

**Ako to funguje:** `requires_grad=True` povie PyTorchu, aby zaznamenával každú operáciu s týmto tenzorom. Počas forward passu si v pamäti postaví rovnaký graf ako v Kroku 1 a každý uzol si pamätá svoje pravidlo pre lokálnu deriváciu. `L.backward()` prejde graf dozadu presne ako Krok 3 a výsledky zapíše do `.grad`.

**Detaily v kóde a prečo:**
- `float64`, aby porovnanie s ručným výsledkom bolo presné a nelíšilo sa na 7. mieste.
- `retain_grad()` na `e` a `d`: PyTorch normálne drží `.grad` len na *listových* tenzoroch (vstupoch, ktoré si vytvoril ty), aby šetril pamäť. Medzivýsledky si vypýtame, aby ich tabuľka mohla ukázať tiež.

---

## Krok 6 · Porovnaj všetky tri a ulož výsledky

`main()` vypíše tabuľku s ručnými, numerickými a torch gradientmi a ✓, keď sa zhodujú (torch vs ručne presne, numericky do 1e-4). Všetko ide do `results.json` a `plot.py` z neho nakreslí graf, dotyčnicu a graf chyby. Výsledok: všade ✓, napr. `∂L/∂a`: −24 / −23,999999999 / −24.

---

## Krok 7 · Zisti, aké malé má byť `h`

```python
def h_sweep():
    exact = manual_grads(**INPUTS)["a"]
    for exp in range(-1, -15, -1):
        h = 10.0 ** exp
        ... abs(numerical_grad("a", h, central=False) - exact) ...
        ... abs(numerical_grad("a", h, central=True) - exact) ...
```

Intuícia hovorí „menšie `h` = presnejšie“. Sweep ukáže, že je to pravda len napoly:

- **Veľké `h`:** dopredná diferencia sa mýli zhruba o 9·h (0,9 pri h = 0,1), lebo ignoruje zakrivenie. Centrálna diferencia je tu už presná, lebo `L(a)` je parabola a symetrická diferencia jej zakrivenie úplne vyruší.
- **Malinké `h` (pod ~1e-8):** obe sa *zhoršia*. `L(a+h)` a `L(a−h)` sú takmer to isté float64 číslo a odčítanie dvoch takmer rovnakých čísel zničí väčšinu platných cifier (**katastrofické vyrušenie**, catastrophic cancellation). Pri h = 1e-14 je chyba ~0,16.

**Ponaučenie:** numerické derivácie sú testovací nástroj s ideálnou zónou (`h ≈ 1e-6` pre float64), nie metóda na tréning.

---

## Vyskúšaj sám

1. Pridaj štvrtú operáciu, napr. `L2 = tanh(L)`, odvoď jej lokálnu deriváciu (`1 − tanh²`) a rozšír všetky tri metódy.
2. Zmeň `b` na `0`. Čo sa stane s `∂L/∂a` a prečo to dáva zmysel podľa pravidla „× vymení vstupy“?
3. Spusti sweep vo `float32` (NumPy `float32`). Kde je nová ideálna zóna pre `h`?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | Vzorec = graf jednoduchých operácií s pomenovanými medzivýsledkami |
| 2 | Gradient = lokálny sklon; krokom proti nemu znižuješ loss |
| 3 | Reťazové pravidlo dozadu: násob lokálne derivácie; `+` kopíruje, `×` vymieňa |
| 4 | Konečné diferencie dávajú nezávislú kontrolu |
| 5 | PyTorch zaznamená graf a spraví ten istý prechod dozadu |
| 7 | Príliš veľké `h` → chyba zakrivenia; príliš malé → chyba zaokrúhlenia |
