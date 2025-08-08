from app import app  # remplace "app" si ton fichier principal a un autre nom

if __name__ == "__main__":
    app.run()

# Ceci permet à gunicorn de voir 'app' comme application Flask
