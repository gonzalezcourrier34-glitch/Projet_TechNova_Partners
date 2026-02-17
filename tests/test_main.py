"""
Il teste la fonction main() de app.main, qui :

lance deux processus (API + dashboard)

surveille leur état

arrête proprement les processus quand il faut

Sans jamais lancer de vrais processus
tout est simulé avec des mocks.

"""
import importlib
import sys
from unittest.mock import MagicMock, patch

# On importe les éléments nécessaires pour tester le module main, 
# notamment pour simuler les processus de l'API et du dashboard.
def reload_main():
    # oublie app.main pour pouvoir le recharger proprement,
    sys.modules.pop("app.main", None)
    # on importe à nouveau le module main, ce qui permet de réinitialiser son état pour chaque test
    import app.main  # noqa
    return importlib.reload(app.main)

# On définit une fonction pour générer un payload d'exemple, 
# qui correspond aux features attendues par le modèle de prédiction.
def test_main_starts_api_and_dashboard_and_stops_dashboard_when_api_exits():
    fake_api = MagicMock()
    fake_dash = MagicMock()

    # API tourne puis s'arrête, dashboard tourne
    fake_api.poll.side_effect = [None, 0]
    fake_dash.poll.return_value = None

    # On patch subprocess.Popen pour retourner nos mocks, 
    # et time.sleep pour éviter de faire attendre le test.
    with patch("app.main.subprocess.Popen") as popen_mock, patch("app.main.time.sleep") as sleep_mock:
        popen_mock.side_effect = [fake_api, fake_dash]
        
        # On recharge le module main pour exécuter la fonction main() avec nos mocks en place, 
        # ce qui permet de tester le comportement de lancement et d'arrêt des processus.
        main = reload_main()
        main.main()

        # On vérifie que les processus ont été lancés et que 
        # le dashboard a été arrêté quand l'API s'est arrêtée, 
        # ce qui correspond au comportement attendu.
        assert popen_mock.call_count == 2
        fake_dash.terminate.assert_called_once()
        fake_api.terminate.assert_not_called()

        # Optionnel: vérifier qu'on a bien "bouclé"
        assert sleep_mock.called

# On teste aussi que si on interrompt le programme avec un KeyboardInterrupt, 
# les deux processus sont correctement terminés, 
# ce qui est important pour éviter de laisser des processus zombies.
def test_main_handles_keyboard_interrupt_and_terminates_both():
    fake_api = MagicMock()
    fake_dash = MagicMock()

    # Les deux processus tournent, puis on simule une interruption clavier.
    fake_api.poll.return_value = None
    fake_dash.poll.return_value = None

    # On patch subprocess.Popen pour retourner nos mocks, et time.sleep pour simuler une interruption clavier, 
    # ce qui permet de tester que les deux processus sont terminés dans ce cas.
    with patch("app.main.subprocess.Popen") as popen_mock, patch(
        "app.main.time.sleep", side_effect=KeyboardInterrupt
    ):
        popen_mock.side_effect = [fake_api, fake_dash]

        main = reload_main()
        main.main()

        fake_api.terminate.assert_called_once()
        fake_dash.terminate.assert_called_once()
