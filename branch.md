# Tutorial Git: lavorare con branch

## 1. Concetti base

### `main`

Branch principale stabile del progetto.

Non dovreste lavorare direttamente qui.

---

### Branch personali / feature branch

Ogni feature o modifica:

* branch separato
* poi merge su `main`

Esempi:

```bash
feature-gnn
feature-dataset
bugfix-parser
```

---

# 2. Clonare la repo

```bash
git clone https://github.com/NOME/repo.git
cd repo
```

---

# 3. Controllare branch attuale

```bash
git branch
```

Quello con `*` è il branch attivo.

---

# 4. Aggiornare `main`

PRIMA di creare un branch:

```bash
git checkout main
git pull origin main
```

Questo:

* entra in main
* scarica aggiornamenti remoti

---

# 5. Creare un branch nuovo

```bash
git checkout -b feature-dataset
```

oppure:

```bash
git switch -c feature-dataset
```

Ora lavori SOLO lì.

---

# 6. Fare modifiche

Crei/modifichi file normalmente.

---

# 7. Controllare modifiche

```bash
git status
```

Ti mostra:

* file modificati
* file non tracciati

---

# 8. Aggiungere file al commit

Tutto:

```bash
git add .
```

Oppure singolo file:

```bash
git add parser.py
```

---

# 9. Fare commit

```bash
git commit -m "Added PGN parser"
```

---

# 10. Push del branch su GitHub

PRIMA volta:

```bash
git push -u origin feature-dataset
```

Dopo basta:

```bash
git push
```

---

# 11. Cambiare branch

```bash
git checkout main
```

oppure:

```bash
git switch main
```

---

# 12. Prendere aggiornamenti da GitHub

```bash
git pull
```

Equivalentemente:

```bash
git pull origin main
```

---

# 13. Merge del branch su main

## Metodo GitHub (consigliato)

1. push branch
2. apri Pull Request
3. merge da GitHub

È il workflow migliore per team.

---

## Metodo terminale

Vai su main:

```bash
git checkout main
```

Aggiorna:

```bash
git pull
```

Merge:

```bash
git merge feature-dataset
```

Push finale:

```bash
git push
```

---

# 14. Eliminare branch vecchio

Locale:

```bash
git branch -d feature-dataset
```

Remoto:

```bash
git push origin --delete feature-dataset
```

---

# Workflow reale in team

## Inizio nuova feature

```bash
git checkout main
git pull
git checkout -b feature-parser
```

---

## Lavori e push

```bash
git add .
git commit -m "Added parser"
git push -u origin feature-parser
```

---

## Aggiornare il proprio branch con `main`

```bash
git checkout main
git pull
```

Poi:

```bash
git checkout feature-parser
git merge main
```

oppure:

```bash
git rebase main
```

---

# Differenza merge vs rebase

## Merge

```bash
git merge main
```

* più semplice
* crea merge commit

---

## Rebase

```bash
git rebase main
```

* storia più pulita
* più professionale
* più rischioso se non sai usarlo

Per iniziare usa:

* `merge`

---

# 15. Risolvere conflitti

Git può mostrare:

```text
CONFLICT
```

Nel file vedrai:

```python
<<<<<<< HEAD
mia versione
=======
altra versione
>>>>>>> main
```

Sistemi manualmente, poi:

```bash
git add .
git commit
```

---

# 16. Comandi importanti

## Vedere log

```bash
git log --oneline --graph
```

---

## Vedere differenze

```bash
git diff
```

---

## Vedere branch

```bash
git branch
```

Remote inclusi:

```bash
git branch -a
```

---

# 17. Errori da evitare

## NON lavorare direttamente su main

---

## NON fare force push a caso

```bash
git push --force
```

può distruggere lavoro altrui.

---

## NON fare commit enormi

Meglio:

* commit piccoli
* commit chiari

---

# 18. Workflow consigliato per il progetto

Branch separati per:

* dataset
* preprocessing
* gnn
* evaluation
* docs

Esempi:

```text
feature-dataset
feature-preprocessing
feature-gnn
feature-eval
```

---

# 19. Proteggere `main`

Su GitHub:

* Settings
* Branches
* Add rule
* Protect `main`

Attiva:

* require pull request
* no direct push

Molto utile nei team.

---

# 20. I 5 comandi fondamentali

```bash
git pull
git checkout -b
git add .
git commit -m
git push
```

Con questi fate già il 90% del workflow team.
