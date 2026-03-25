# AlphaZero Parallel for Connect Four and Frequency Assignment

Acest repository conține o implementare educațională a algoritmului **AlphaZero** și a variantei sale adaptate pentru o problemă de optimizare din telecomunicații: **Frequency Assignment Problem (FAP)**. Proiectul este construit în notebook-uri Jupyter și combină:

- **Monte Carlo Tree Search (MCTS)**
- **self-play**
- **rețele neuronale în PyTorch**
- **antrenare paralelizată**
- **accelerare pe GPU AMD prin DirectML**

## Conținutul repository-ului

### `AlphaZeroParallel.ipynb`
Notebook-ul principal pentru **Connect Four**.

Include:
- definirea mediului de joc `ConnectFour`
- arhitectura rețelei `ResNet` cu blocuri reziduale
- implementarea `MCTSParallel` pentru explorare în paralel
- clasa `AlphaZeroParallel` pentru self-play și antrenare
- salvarea modelelor antrenate (`model_*`, `optimizer_*`)
- o secțiune finală de evaluare și vizualizare a politicii modelului

### `FAPParallel.ipynb`
Extinderea ideii AlphaZero către **Frequency Assignment Problem**.

Include:
- mediul `FrequencyEnvironment`, unde turnurile trebuie să primească frecvențe
- o matrice de interferență calculată din pozițiile turnurilor
- o rețea de tip `MLP` pentru policy + value
- căutare de tip MCTS adaptată pentru problemă single-agent
- antrenare paralelizată pentru instanțe mici și apoi pentru un scenariu scalat
- evaluare pe:
  - **7 turnuri / 3 frecvențe**
  - **30 turnuri / 4 frecvențe**

### `PrezentareAlphaZero.ipynb`
Notebook orientat spre **explicație și prezentare**.

Include:
- introducere în reinforcement learning
- AlphaZero pentru jocuri zero-sum (`TicTacToe`, `ConnectFour`)
- descrierea arhitecturii `ResNet`
- explicația MCTS
- adaptarea ideii la FAP
- partea de scalabilitate pentru rețele mai mari

Acest notebook este util dacă vrei să înțelegi logica proiectului înainte să rulezi experimentele.

## Obiectivul proiectului

Scopul proiectului este să arate că aceeași idee generală — **policy/value network + MCTS + self-play / search-guided learning** — poate fi folosită atât pentru:

1. **jocuri competitive** precum Connect Four
2. **probleme de optimizare secvențială** precum alocarea frecvențelor în telecomunicații

Astfel, repository-ul are atât valoare didactică, cât și valoare demonstrativă pentru aplicarea AI în probleme reale.

## Arhitectură

### 1. Connect Four
Pipeline-ul este:

1. se generează stări de joc prin self-play
2. MCTS explorează mutările promițătoare
3. rețeaua `ResNet` produce:
   - **policy head**: probabilitățile mutărilor
   - **value head**: evaluarea poziției
4. memoria colectată este folosită pentru antrenarea modelului

### 2. Frequency Assignment Problem
Pipeline-ul este similar, dar adaptat pentru o problemă single-agent:

1. starea conține turnul curent și frecvențele deja alocate
2. mediul calculează penalizarea din interferențe
3. rețeaua `FAP_MLP` estimează politica și valoarea stării
4. MCTS alege frecvențele care duc la interferență mai mică

## Tehnologii folosite

- **Python**
- **Jupyter Notebook**
- **NumPy**
- **PyTorch**
- **Matplotlib**
- **tqdm**
- **torch-directml** (pentru GPU AMD, opțional)

## Cerințe

Poți instala dependențele de bază astfel:

```bash
pip install numpy matplotlib tqdm torch
```

Pentru rulare pe GPU AMD cu DirectML:

```bash
pip install torch-directml
```

> Dacă `torch-directml` nu este disponibil, notebook-urile cad automat pe CPU.

## Cum rulezi proiectul

### Varianta 1: Jupyter Notebook

```bash
jupyter notebook
```

Apoi deschizi pe rând:
- `PrezentareAlphaZero.ipynb`
- `AlphaZeroParallel.ipynb`
- `FAPParallel.ipynb`

### Varianta 2: Google Colab / mediu compatibil notebook

Poți încărca notebook-urile direct și rula celulele secvențial.

## Fișiere generate

În timpul antrenării pentru Connect Four, notebook-ul poate salva:

- `model_<iteratie>_ConnectFour.pt`
- `optimizer_<iteratie>_ConnectFour.pt`

Aceste fișiere pot fi apoi încărcate pentru evaluare și inferență.

## Idei importante din implementare

### Self-play în paralel
Codul folosește mai multe jocuri simultan (`num_parallel_games`) pentru a accelera colectarea de date, ceea ce este util mai ales în notebook-uri și pe GPU.

### Explorare controlată
Se folosesc:
- **Dirichlet noise** la rădăcina arborelui
- un număr variabil de căutări (`num_searches` vs `num_searches_fast`)
- temperatură pentru controlul explorării în timpul self-play-ului

### Adaptare către telecomunicații
Partea de FAP este interesantă deoarece transformă o problemă de alocare într-o decizie secvențială, unde modelul învață să minimizeze interferența prin search ghidat de rețea.

## Puncte forte

- proiectul arată o înțelegere bună a mecanismului AlphaZero
- există o legătură clară între teorie și aplicație practică
- implementarea pentru FAP este o extensie originală și relevantă
- notebook-ul de prezentare face proiectul ușor de urmărit într-un context academic
- suportul pentru DirectML îl face accesibil și pe sisteme cu GPU AMD

## Limitări curente

Repository-ul este funcțional și bun pentru demonstrație, dar poate fi îmbunătățit pentru utilizare mai matură:

- codul este duplicat între notebook-uri
- logica este concentrată în notebook-uri, nu în module Python separate
- lipsesc un `requirements.txt` și o structură de pachet clară
- unele valori și căi sunt hardcodate
- nu există un sistem de configurare separat pentru hiperparametri
- antrenarea, evaluarea și vizualizarea sunt amestecate în aceleași notebook-uri

## Direcții de îmbunătățire

- mutarea codului comun în fișiere `.py` (`games.py`, `models.py`, `mcts.py`, `train.py`)
- adăugarea unui `requirements.txt`
- separarea clară între training și evaluation
- salvarea automată a metricilor (loss, win-rate, reward, interference)
- adăugarea unui script dedicat pentru inferență
- documentarea experimentelor și a rezultatelor obținute

## Cui i se adresează

Acest proiect este potrivit pentru:

- proiecte academice de AI / ML
- demonstrații de AlphaZero și MCTS
- explorarea RL în probleme non-game
- prezentări tehnice sau de licență / disertație

## Concluzie

Repository-ul demonstrează bine cum poate fi extinsă filosofia AlphaZero din zona jocurilor către o problemă inginerească reală. Dincolo de valoarea educațională, proiectul sugerează o direcție interesantă: folosirea search-ului și a rețelelor neuronale pentru probleme de optimizare combinatorială.

---
