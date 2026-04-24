# Update-Notizen & Fehlerbehebungen

### Build-Prozess & Docker Setup
- `make dev-build` und `make dev-up` wurden getrennt und um Argumente erweitert.
- `DEVICE=cu121 make dev-build` ausgeführt (Dauer bei einer 250-Mbit-Leitung: 534s).
- Profile zur `docker-compose.dev.yaml` hinzugefügt (Training, Finetuning und Evaluation werden durch Airflow gestartet).

### Fehlerbehebungen bei `make dev-up`
- **Nginx:** Ordner und Config fehlten noch im Main (Config manuell hinzugefügt).
- **Monitoring:** `monitoring/prometheus.yml` existiert nicht mit der Endung `.yml`, sondern nur als `.yaml` (Docker Compose angepasst).
- **BentoML:** - Kann initial nicht gestartet werden, da es erst mit `make containerize-bento` gebaut werden muss. (Sollte in den Build-Prozess integriert werden).
    - Namenskonvention: Bindestriche statt Unterstriche verwendet. Anpassung der `bentofile.yaml`, `docker-compose` sowie der `bento-service.py`.
    - Makefile korrigiert: Der `latest`-Tag für den BentoML-Container fehlte. Jetzt: `rakuten-text-service:latest -t rakuten-ml/rakuten-text-service:latest`.

### Infrastruktur & Airflow
- Probleme mit der aktuellen `docker-compose.infrastructure.yaml` (Airflow startet nicht richtig). Alte Version verwendet.
- **Wichtig:** `AIRFLOW_CONN_FS_DEFAULT: 'fs://?path=%2F'` muss zu den Umgebungsvariablen hinzugefügt werden!
- Mismatch bei `bentoml.yaml` und `pyproject.toml` korrigiert. `pyproject.toml` bereinigt und Image entfernt.
- **Daten-Hinweis:** Wenn sich Raw-Daten geändert haben und noch nicht gepusht wurden, schlägt der `train-text-run` fehl.

### Airflow Permissions & Environment
- Airflow-Berechtigungen fehlten; ein Setup-Script wird benötigt.
- Temporärer Fix für Permission-Probleme: `sudo chown -R 1000:0 ./data ./configs ; sudo chmod -R 775 ./data ./configs`.
- *Hinweis:* Das ist keine dauerhafte Lösung (siehe lokale Konfiguration bei Eduard, welche Ordner den Owner `50000` haben).
- Um Docker auszuführen, muss `group_add: -> "${DOCKER_GID}"` hinzugefügt werden (siehe `setup.sh`).
- **Device-Konfiguration:** In der `.env` befindet sich die Config für das Airflow-DEVICE. Airflow muss mitgeteilt bekommen, mit welchem Device das Retraining starten soll.
- **Mounting:** `/app` existiert im Airflow-Container, aber Docker mountet vom HOST, nicht aus dem Container. Daher `PROJECT_ROOT=.` in der `.env`.
- Details siehe `dags/retrain_new_data.py`.

### Sonstiges
- Die `.env` wurde eventuell versehentlich überschrieben. Falls das System stabil läuft, ist dies kein Problem.




- Wichtig make dev-build und dev-up beiden mit DEVICE=cu121 starten, wenn auf GPU trainiert werden soll.
- GIT_TOKEN wurde hinzugefügt damit airflow user aus dem airflow pushen kann (dvc/git/dagshub)
