# Ce qui a changé et pourquoi (version 2)

Ce fichier explique chaque changement : **quoi**, **pourquoi** (quelle exigence de l'énoncé)
et **comment le montrer** pendant la démo. À la fin : les questions probables du jury.

---

## Vue d'ensemble

| Exigence de l'énoncé | Avant | Maintenant | Fichier(s) |
|---|---|---|---|
| Étape 2 : « zone dwell time » + temps d'attente (Queue Wait-Time) | absent | événement `dwell` à chaque sortie de zone | `vision_common.py` (`DwellTracker`), `step2_telemetry.py` |
| Étape 2 : « defect/stockout alerts » + alertes rupture de stock | absent | caméra rayon (option `--shelf`) → événement `stockout` + image | `step2_telemetry.py` |
| Étape 2 : télémétrie **structurée** | `event: dict`, `track_ids` perdus | validation Pydantic (422), 4 types d'événements, `track_ids` stockés | `api.py`, `telemetry_db.py` |
| App exemple : « logs structured **hourly KPIs** into a BI dashboard » | calcul à l'affichage seulement | table `kpi_hourly` + `GET /kpi/hourly` + dashboard | `telemetry_db.py`, `step5_dashboard.py` |
| App exemple : « tracks unique customer **trajectories** across store zones » | comptage seulement | `GET /trajectories` : parcours de chaque ID entre zones | `telemetry_db.py`, `api.py` |
| Étape 4 : « **attaching the exact annotated frame** » | seulement le chemin du fichier | l'image JPEG elle-même (base64) + son SHA-256 | `step4_alerts.py` |
| Étape 4 : « **deterministic ByteTrack object count** » | `count` seul | `count` + `track_ids` (count = nombre d'IDs) | `step4_alerts.py` |
| Étape 4 : « **Verification Gate** » | absent | nœud n8n qui vérifie l'alerte, 200 ou 422 | `n8n/vision_alert_workflow.json` |
| Diagramme : « OpenCV **Frame Buffer** » (architecture découplée) | lecture et inférence dans la même boucle | thread de capture + file d'attente bornée | `vision_common.py` (`FrameBuffer`) |
| Paradigme : « Detection **& Segmentation** » | `yolov8n-seg.pt` jamais utilisé | option `--seg` (masques) | `step1_tracking.py` |
| Slide 15 : `python check_environnement.py` | le script n'existait pas | créé | `check_environnement.py` |
| Streamlit (outil imposé) | non mentionné dans la présentation, absent du Pipfile | dashboard complet + dépendances | `step5_dashboard.py`, `Pipfile`, `requirements.txt` |

---

## 1. Bug corrigé : la vidéo tourne à 12 images/s, le code supposait 30

`data/people.mp4` = 768×432, **12 fps**, 596 images (≈ 50 s). Trois valeurs étaient donc fausses :

| Réglage | Avant | Effet réel à 12 fps | Maintenant |
|---|---|---|---|
| `sv.ByteTrack()` sans `frame_rate` | piste perdue gardée 30 images | 2,5 s (la slide 9 dit 1 s) | `make_tracker(fps)` → **1 s** |
| `EMIT_EVERY_N_FRAMES = 30` | 1 événement / 30 images | 1 toutes les 2,5 s (les slides disent ≈ 1/s) | `EMIT_EVERY_SECONDS = 1.0` → **1 par seconde** |
| `SUSTAIN_FRAMES = 10` | 10 images | 0,83 s (les slides disent 5 s) | `--sustain-seconds 1.0` → **1 s = 12 images** |

**À retenir :** tout est maintenant exprimé en **secondes** et converti avec le vrai fps de la vidéo.
Dans supervision, la durée de garde d'une piste perdue = `lost_track_buffer × frame_rate / 30` images.

---

## 2. Fichiers partagés (nouveau)

Avant, `ZONES` et toute la configuration étaient copiés dans 4 scripts. Maintenant :

- **`vision_common.py`** (côté *edge*) : zones, tracker, `DwellTracker`, `FrameBuffer`, encodage d'image.
- **`telemetry_db.py`** (côté *cloud*) : tables SQLite, agrégation des KPI horaires, trajectoires.
- **`api.py`** (côté *cloud*) : l'API FastAPI avec validation Pydantic.

Chaque `stepX.py` reste un script court qui se lit de haut en bas.

**Zones :** la `queue zone` est maintenant un **trapèze** (plus large en bas) pour suivre la
perspective du sol : c'est un vrai polygone personnalisé, pas un rectangle.

---

## 3. Temps passé dans une zone (dwell time) = temps d'attente

**Classe `DwellTracker`** (`vision_common.py`). Pour chaque couple (zone, ID) elle retient
la première et la dernière image où la personne est dans la zone.

- La visite se termine quand l'ID est absent depuis plus de **0,5 s** (une personne cachée
  une fraction de seconde ne coupe pas sa visite en deux).
- La durée est calculée en **temps vidéo** (`numéro d'image / fps`), pas avec l'horloge :
  le résultat ne dépend pas de la vitesse du PC.
- À la sortie, `step2_telemetry.py` envoie :
  ```json
  {"event_type": "dwell", "zone_name": "queue zone", "track_id": 6,
   "dwell_seconds": 3.67, "details": {"entered_s": 30.1, "exited_s": 33.8}}
  ```
- Dans `step1_tracking.py`, l'étiquette affiche le temps en direct : `#3 2.4s`.

**Démo :** lancer `step2_telemetry.py` → lignes `dwell: #6 stayed 3.7s in queue zone`,
puis `http://127.0.0.1:8000/events?event_type=dwell`.

---

## 4. Rupture de stock (stockout)

**Option `--shelf` de `step2_telemetry.py`** = une deuxième caméra, tournée vers un rayon
(même script que la télémétrie : l'API est lancée une seule fois et l'événement `stockout`
porte le même `run_id`).

1. YOLO compte les objets « produits » de COCO dans la zone rayon : bouteille (39), tasse (41),
   bol (45), banane (46), pomme (47), orange (49), carotte (51).
2. Si le compte < `--min-stock` (sur une image, ou 3 vérifications de suite sur une vidéo),
   → image annotée `output/stockout_<date>.jpg` + événement `stockout` dans l'API
   (+ webhook n8n si `--webhook` est donné).
3. Une seule alerte par épisode ; elle se réarme quand le rayon est rempli.

**Limite honnête (à dire au jury) :** le modèle COCO ne connaît que quelques produits.
Dans un vrai magasin, on ré-entraîne YOLO sur ses propres produits (ex. jeu de données SKU-110K) ;
le pipeline (compter dans une zone → seuil → événement + image) reste identique.

**Pour la démo :** il vous faut une photo de rayon (bouteilles, fruits…), idéalement deux :
un rayon plein et un rayon presque vide.
```
python step2_telemetry.py --shelf data/rayon_plein.jpg --min-stock 5    -> pas d'alerte
python step2_telemetry.py --shelf data/rayon_vide.jpg  --min-stock 5    -> STOCKOUT
```

---

## 5. API structurée (FastAPI + Pydantic)

**Avant :** `def add_event(event: dict)` → un JSON incomplet provoquait une erreur 500, et les
`track_ids` étaient reçus mais jamais stockés.

**Maintenant (`api.py`) :** un modèle `Event` Pydantic avec 4 types :

| `event_type` | Champs obligatoires | Règle vérifiée |
|---|---|---|
| `zone_count` | `zone_name`, `object_count`, `track_ids` | `object_count == len(track_ids)` |
| `dwell` | `zone_name`, `track_id`, `dwell_seconds ≥ 0` | |
| `stockout` | `zone_name`, `object_count` | |
| `alert` | `zone_name`, `object_count`, `track_ids` | `object_count == len(track_ids)` |

Un événement incorrect → **HTTP 422** avec la raison (visible dans `/docs`).

**Base de données :** la table `events` gagne les colonnes `event_type`, `track_ids` (JSON),
`track_id`, `dwell_seconds`, `run_id`, `details`. Une ancienne base est **mise à niveau
automatiquement** (aucune donnée perdue). Conseil : supprimez `data/events.db` avant la démo
pour n'avoir que les données du jour.

**`run_id` :** ByteTrack recommence à `#1` à chaque lancement ; sans `run_id`, le `#3` de lundi et
le `#3` de mardi seraient comptés comme la même personne.

**Endpoints :**

| Endpoint | Rôle |
|---|---|
| `POST /events` | recevoir un événement validé |
| `GET /events?event_type=dwell&limit=20` | derniers événements |
| `GET /kpi/hourly` | reconstruit et renvoie la table `kpi_hourly` |
| `GET /trajectories` | parcours de chaque personne entre les zones |

---

## 6. KPI horaires (table `kpi_hourly`)

Une ligne par (heure, zone), recalculée à partir des événements :

| Colonne | Signification |
|---|---|
| `avg_count`, `peak_count` | occupation moyenne et maximale |
| `unique_visitors` | nombre d'IDs différents vus dans la zone (par `run_id`) |
| `visits`, `avg_dwell_s`, `max_dwell_s` | passages et temps d'attente moyen / max |
| `alerts`, `stockouts` | nombre d'alertes d'affluence et de ruptures |

Exemple obtenu sur `people.mp4` (test) : queue zone → occupation moyenne 0,26, pic 2,
7 visiteurs uniques, attente moyenne 2,15 s, max 3,67 s.

---

## 7. Trajectoires entre zones

`GET /trajectories` regroupe les événements `dwell` par personne et les trie dans le temps :
```json
{"track_id": 1, "path": ["queue zone", "entrance zone"], "total_dwell_s": 3.33}
```
C'est le « tracks unique customer trajectories across store zones » de l'énoncé.

---

## 8. Étape 4 : l'image est vraiment envoyée + vérification

**Avant :** le message contenait `frame_path: "output/alert_frame_....jpg"`, un chemin sur votre
PC. n8n ne pouvait pas lire ce fichier s'il tournait sur une autre machine.

**Maintenant, le message contient :**
```json
{"alert_type": "zone_occupancy", "zone": "queue zone",
 "count": 2, "track_ids": [3, 2], "threshold": 2,
 "sustained_frames": 12, "sustain_required_frames": 12,
 "frame_index": 210, "video_time_s": 17.5,
 "image_base64": "/9j/4AAQ...", "image_sha256": "61671a1a...", ...}
```
- `image_base64` : l'image JPEG elle-même ; `image_sha256` : son empreinte (preuve que
  l'image reçue est exactement celle sauvegardée).
- `track_ids` : le compte est **déterministe** = exactement ces IDs ByteTrack, visibles sur l'image.
- Un bandeau rouge est écrit sur l'image : `ALERT queue zone: 2 >= 2  IDs [3, 2]`.
- L'alerte est aussi stockée dans l'API (`event_type: alert`) si elle tourne.

**Workflow n8n (`n8n/vision_alert_workflow.json`, à importer) :**
```
Webhook (POST /vision-alert)
  → Verification gate (Code)
  → Verified? (IF)
       oui → Save image → Respond 200 {status: accepted, message}
       non → Respond 422 {status: rejected, reasons}
```
La « Verification gate » vérifie : image présente, `count == len(track_ids)`,
`count >= threshold`, règle maintenue assez longtemps (et pour un stockout : `count < minimum`).
Elle prépare aussi le message lisible, par ex.
« Affluence - queue zone : 2 personnes (IDs 3, 2), seuil 2, maintenu 12 images ».

**Vérification à deux niveaux :** en Python (seuil tenu 1 s, une alerte par épisode) puis dans n8n.

---

## 9. Frame Buffer découplé

`FrameBuffer` (utilisé par l'étape 2) : un **thread** lit la vidéo et remplit une **file bornée**
(64 images) ; le worker YOLO lit dans cette file.
- Fichier vidéo : si la file est pleine, la lecture attend → aucune image perdue.
- Flux en direct (webcam/RTSP) : si la file est pleine, on jette la plus ancienne → le worker
  traite toujours des images récentes (temps réel).

C'est la boîte « OpenCV Frame Buffer » du diagramme de l'énoncé.

---

## 10. Autres changements

- **`step1_tracking.py --seg`** : modèle `yolov8n-seg.pt` + masques (`sv.MaskAnnotator`).
- **Heatmap (étape 3)** : utilise la position des **pieds** (bas-centre de la boîte) au lieu du
  centre, comme les zones → la carte montre où les gens se tiennent au sol.
- **`check_environnement.py`** : versions des bibliothèques, modèles, vidéo (et son fps).
- **Dashboard** : KPI par zone (dont temps moyen et visiteurs uniques), table horaire,
  temps par zone, trajectoires, alertes et ruptures, images de preuve, heatmap.
- **Dépendances** : `streamlit` et `pandas` ajoutés au `Pipfile` ; `onnx`, `pandas` à
  `requirements.txt`. Après mise à jour du Pipfile : `pipenv lock` puis `pipenv install`.
- **Docs** : README, RUN.md (étapes 4 à 10) et ARCHITECTURE.md à jour.

---

## Ce qui a été testé

Les bibliothèques ultralytics / supervision / fastapi n'étaient pas installables dans mon
environnement de test. J'ai donc exécuté **vos scripts tels quels** sur `data/people.mp4` avec des
remplaçants : le **vrai modèle `yolov8n.onnx`** (onnxruntime) pour la détection, et un tracker IoU
simplifié à la place de ByteTrack. Résultats vérifiés :
- étape 2 : 596 images, 111 événements, 11 temps de passage, KPI horaires, 10 trajectoires ;
  mise à niveau automatique de votre ancienne `events.db` ;
- étape 4 : 2 alertes ; l'image décodée du message est identique au fichier (SHA-256 vérifié) ;
- le code de la « Verification gate » n8n exécuté avec Node.js sur ce message : accepté ;
  avec un message faux : rejeté avec les raisons ;
- Pydantic : 4 événements incorrects → 422 ;
- étapes 1, 2 (dont `--shelf`), 3 et fonctions du dashboard : exécutées sans erreur.

**À faire chez vous** (avec les vraies bibliothèques) : lancer chaque étape une fois
(`RUN.md`), importer le workflow dans n8n et tester avec `/webhook-test/vision-alert`.
Les identifiants exacts (#2, #3…) peuvent différer un peu, car le vrai ByteTrack n'est pas
exactement le tracker utilisé pour mes tests.

---

## Questions probables du jury (et réponses courtes)

- **Pourquoi le temps vidéo et pas l'horloge pour le dwell time ?** Parce qu'il ne dépend pas de
  la vitesse du CPU : 12 images à 12 fps = 1 s, même si le PC traite à 5 fps.
- **Pourquoi envoyer l'image en base64 et pas le chemin ?** Le chemin n'existe que sur la machine
  edge ; n8n peut être ailleurs. Le SHA-256 prouve que l'image n'a pas été modifiée.
- **Que veut dire « count déterministe » ?** Le compte est exactement la liste des IDs ByteTrack
  dans la zone sur l'image envoyée : on peut recompter les boîtes sur la photo.
- **Pourquoi SQLite et pas PostgreSQL ?** Prototype en un fichier sans serveur. Le SQL est
  standard : en production on passe à PostgreSQL/TimescaleDB en changeant la connexion.
- **Où est l'Event Bus ?** Dans le prototype, c'est `POST /events` (HTTP). En production :
  Kafka, Redis Streams ou MQTT, pour absorber les pics et rejouer les événements.
- **La rupture de stock marche vraiment ?** Le pipeline oui ; la précision dépend du modèle.
  Avec COCO : bouteilles, fruits, tasses. Pour un vrai magasin : YOLO ré-entraîné sur ses produits.
- **Pourquoi garder la piste 1 s ?** Pour qu'une personne cachée brièvement garde son ID ; plus
  long, on risque de donner l'ID à quelqu'un d'autre.
