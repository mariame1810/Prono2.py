from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class RobotPredictor:
    def __init__(self, equipe_A, equipe_B, scores_passes, confrontations_directes, forme_equipe):
        self.equipe_A = equipe_A
        self.equipe_B = equipe_B
        self.scores_passes = scores_passes
        self.confrontations_directes = confrontations_directes
        self.forme_equipe = forme_equipe
        self.model_regress_lin = LinearRegression()
        self.model_knn = KNeighborsRegressor(n_neighbors=3)  # Utilisons k=3 pour le k-NN
        self.model_random_forest = RandomForestRegressor(n_estimators=100)  # Nombre d'arbres = 100
        self.model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
        self.model_gradient_boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        self.model_bagging = BaggingRegressor(n_estimators=100)

    def preparer_donnees_entree(self):
        X = []
        for i, (score, confrontations, forme) in enumerate(zip(self.scores_passes, self.confrontations_directes, self.forme_equipe)):
            features = [i+1]  # Numéro du match comme première feature
            features.extend(score)  # Ajouter les scores passés
            features.extend(confrontations)  # Ajouter l'historique des confrontations directes
            features.extend(forme)  # Ajouter les statistiques de forme récente de chaque équipe
            X.append(features)
        return X

    def entrainer_modele_regress_lin(self, X):
        self.model_regress_lin.fit(X, self.scores_passes)

    def entrainer_modele_knn(self, X):
        self.model_knn.fit(X, self.scores_passes)

    def entrainer_modele_random_forest(self, X):
        y = [max(score) for score in self.scores_passes]
        self.model_random_forest.fit(X, self.scores_passes)

    def entrainer_modele_nn(self, X):
        self.model_nn.fit(X, self.scores_passes)

    def entrainer_modele_gradient_boosting(self, X):
        y = np.mean(self.scores_passes, axis=1)
        self.model_gradient_boosting.fit(X, y)

    def entrainer_modele_bagging(self, X):
        self.model_bagging.fit(X, self.scores_passes)

    def faire_prediction_regress_lin(self):
        # Utiliser les scores passés pour entraîner le modèle de régression linéaire
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        y = self.scores_passes
        self.entrainer_modele_regress_lin(X)

        # Prédire le prochain score
        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_regress_lin.predict([[prochain_numero_match]])

        return prediction[0]

    def faire_prediction_knn(self):
        # Utiliser les scores passés pour entraîner le modèle k-NN
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        y = self.scores_passes
        self.entrainer_modele_knn(X)

        # Prédire le prochain score avec le modèle k-NN
        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_knn.predict([[prochain_numero_match]])

        return prediction[0]

    def faire_prediction_moyenne(self):
        moyenne = np.mean(self.scores_passes, axis=0)  # Calculer la moyenne pour chaque élément du couple
        return moyenne

    def faire_prediction_mediane(self):
        mediane = np.median(self.scores_passes, axis=0)  # Calculer la médiane pour chaque élément du couple
        return mediane

    def faire_prediction_minimum_exact(self):
        # Prédire le minimum de buts basé sur les scores exacts
        minimum_exact = np.min(self.scores_passes, axis=0)
        return minimum_exact

    def faire_prediction_random_forest(self):
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        self.entrainer_modele_random_forest(X)

        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_random_forest.predict([[prochain_numero_match]])

        return prediction[0]


    def faire_prediction_nn(self):
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        self.entrainer_modele_nn(X)

        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_nn.predict([[prochain_numero_match]])

        return prediction[0]

    def faire_prediction_gradient_boosting(self):
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        self.entrainer_modele_gradient_boosting(X)

        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_gradient_boosting.predict([[prochain_numero_match]])

        return prediction[0], prediction[0]  # Retourner le même score pour les deux équipes


    def faire_prediction_bagging(self):
        X = [[i] for i in range(1, len(self.scores_passes) + 1)]
        self.entrainer_modele_bagging(X)

        prochain_numero_match = len(self.scores_passes) + 1
        prediction = self.model_bagging.predict([[prochain_numero_match]])

        return prediction[0]

    def determiner_resultat(self, prediction):
        # Comparer la prédiction pour déterminer le résultat du match
        seuil_victoire = 0.5  # ajusté en fonction du fait que le score est maintenant un couple
        seuil_nul = 0.5  # ajusté en fonction du fait que le score est maintenant un couple

        if len(prediction) == 1:  # Cas où la prédiction est une seule valeur scalaire
            if prediction[0] > seuil_victoire:
                return f"Le vainqueur sera {self.equipe_A}"
            elif prediction[0] < -seuil_victoire:
                return f"Le vainqueur sera {self.equipe_B}"
            else:
                return "Le match sera nul"
        elif len(prediction) == 2:  # Cas où la prédiction est un couple de valeurs
            if prediction[0] > prediction[1] + seuil_victoire:
                return f"Le vainqueur sera {self.equipe_A}"
            elif prediction[1] > prediction[0] + seuil_victoire:
                return f"Le vainqueur sera {self.equipe_B}"
            else:
                return "Le match sera nul"
        else:
            return "Erreur : La prédiction n'est pas dans un format attendu"

    def calculer_scores_passes_inverse(self, resultat):
        # Calculer les scores passés en fonction du résultat du match
        if "vainqueur" in resultat.lower():
            # Si l'équipe A est le vainqueur
            scores_passes_inverse = [tuple(score + 1 for score in couple) for couple in self.scores_passes]
        elif "nul" in resultat.lower():
            # Si le match est nul
            scores_passes_inverse = self.scores_passes
        else:
            # Si l'équipe B est le vainqueur
            scores_passes_inverse = [tuple(score - 1 for score in couple) for couple in self.scores_passes]

        return scores_passes_inverse

    def evaluer_methodes(self):
        # Évaluer les méthodes de calcul et déterminer la plus précise
        prediction_regress_lin = self.faire_prediction_regress_lin()
        prediction_knn = self.faire_prediction_knn()
        prediction_moyenne = self.faire_prediction_moyenne()
        prediction_mediane = self.faire_prediction_mediane()
        prediction_minimum_exact = self.faire_prediction_minimum_exact()
        prediction_random_forest = self.faire_prediction_random_forest()
        prediction_nn = self.faire_prediction_nn()
        prediction_gradient_boosting = self.faire_prediction_gradient_boosting()
        prediction_bagging = self.faire_prediction_bagging()


        resultats = {
            'Régression Linéaire': (prediction_regress_lin, self.determiner_resultat(prediction_regress_lin)),
            'k-NN': (prediction_knn, self.determiner_resultat(prediction_knn)),
            'Moyenne des Scores Passés': (prediction_moyenne, self.determiner_resultat(prediction_moyenne)),
            'Médiane des Scores Passés': (prediction_mediane, self.determiner_resultat(prediction_mediane)),
            'Minimum Exact des Scores Passés': (prediction_minimum_exact, self.determiner_resultat(prediction_minimum_exact)),
            'Forêts d\'arbres décisionnels': (prediction_random_forest, self.determiner_resultat(prediction_random_forest)),
            'Réseaux de neurones': (prediction_nn, self.determiner_resultat(prediction_nn)),
            'Gradient Boosting': (prediction_gradient_boosting, self.determiner_resultat(prediction_gradient_boosting)),
            'Bagging': (prediction_bagging, self.determiner_resultat(prediction_bagging))
        }

        # Calculer l'accuracy pour chaque méthode (le nombre de résultats corrects)
        accuracies = {methode: sum(np.array_equal(resultats[methode][0], resultats[m][0]) for m in resultats) for methode in resultats}

        # Trouver la méthode la plus précise
        methode_precise = max(accuracies, key=accuracies.get)

        return resultats, methode_precise

    def obtenir_scores_predits(self):
    # Obtenir les scores prédits selon chaque méthode de calcul
        scores_predits = {
            'Régression Linéaire': self.faire_prediction_regress_lin(),  # Modifier cette ligne
            'k-NN': self.faire_prediction_knn(),  # Modifier cette ligne
            'Moyenne des Scores Passés': self.faire_prediction_moyenne(),
            'Médiane des Scores Passés': self.faire_prediction_mediane(),
            'Minimum Exact des Scores Passés': self.faire_prediction_minimum_exact(),
            'Forêts d\'arbres décisionnels': self.faire_prediction_random_forest(),
            'Réseaux de neurones': self.faire_prediction_nn(),
            'Bagging': self.faire_prediction_bagging(),
            'Gradient Boosting': self.faire_prediction_gradient_boosting()
        }

        return scores_predits


# Exemple d'utilisation avec les nouveaux paramètres
equipe_A = "Équipe A"
equipe_B = "Équipe B"

# Scores passés, confrontations directes et forme récente des équipes (exemples)
scores_passes_manuels = [
    (2, 2), (1, 3), (1, 1), (2, 2), (1, 2),
    (0, 1), (1, 1), (0, 1), (1, 1), (3, 1)
]
confrontations_directes = [
    (0, 1), (0, 0), (0, 2), (1, 0), (0, 1),
    (0, 2), (1, 1), (1, 0), (1, 1), (0, 0)
]
#en gros cette saison l'equipe A a gagné x fois face à la B et la B y fois face à la A (x,y) = tuple de nombre de matchs gagnés de A et B face à l'un et l'autre sur une saison
forme_equipe = [
    (4, 5), (4, 5), (1, 2), (8, 4), (4, 1),
    (4, 6), (3, 2), (2, 3), (3, 4), (6, 2)
]

# Création d'une instance de RobotPredictor
robot_predictor = RobotPredictor(equipe_A, equipe_B, scores_passes_manuels, confrontations_directes, forme_equipe)

# Préparation des données d'entrée
X = robot_predictor.preparer_donnees_entree()

# Entraînement des modèles
robot_predictor.entrainer_modele_regress_lin(X)
robot_predictor.entrainer_modele_knn(X)
robot_predictor.entrainer_modele_random_forest(X)
robot_predictor.entrainer_modele_nn(X)
robot_predictor.entrainer_modele_gradient_boosting(X)
robot_predictor.entrainer_modele_bagging(X)

# Afficher les scores passés saisis manuellement
print("Scores passés saisis manuellement :")
for score in scores_passes_manuels:
    print(f"({score[0]}-{score[1]})")

# Prédiction avec la méthode de la moyenne
prediction_moyenne = robot_predictor.faire_prediction_moyenne()
resultat_moyenne = robot_predictor.determiner_resultat(prediction_moyenne)
print(f"\nPrédiction (Moyenne) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_moyenne}")

# Prédiction avec la méthode de la médiane
prediction_mediane = robot_predictor.faire_prediction_mediane()
resultat_mediane = robot_predictor.determiner_resultat(prediction_mediane)
print(f"Prédiction (Médiane) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_mediane}")

# Prédiction avec la méthode de la régression linéaire
prediction_regress_lin = robot_predictor.faire_prediction_regress_lin()
resultat_regress_lin = robot_predictor.determiner_resultat(prediction_regress_lin)
print(f"Prédiction (Régression Linéaire) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_regress_lin}")

# Prédiction avec la méthode du k-NN
prediction_knn = robot_predictor.faire_prediction_knn()
resultat_knn = robot_predictor.determiner_resultat(prediction_knn)
print(f"Prédiction (k-NN) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_knn}")

# Prédiction avec la méthode du minimum exact des scores passés
prediction_minimum_exact = robot_predictor.faire_prediction_minimum_exact()
resultat_minimum_exact = robot_predictor.determiner_resultat(prediction_minimum_exact)
print(f"Prédiction (Minimum Exact) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_minimum_exact}")

# Prédiction avec la méthode des forêts d'arbres décisionnels
prediction_random_forest = robot_predictor.faire_prediction_random_forest()
resultat_random_forest = robot_predictor.determiner_resultat(prediction_random_forest)
print(f"Prédiction (Forêts d'arbres décisionnels) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_random_forest}")

# Prédiction avec la méthode des réseaux de neurones
prediction_nn = robot_predictor.faire_prediction_nn()
resultat_nn = robot_predictor.determiner_resultat(prediction_nn)
print(f"Prédiction (Réseaux de neurones) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_nn}")

# Prédiction avec la méthode du gradient boosting
prediction_gradient_boosting = robot_predictor.faire_prediction_gradient_boosting()
resultat_gradient_boosting = robot_predictor.determiner_resultat(prediction_gradient_boosting)
print(f"Prédiction (Gradient Boosting) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_gradient_boosting}")

# Prédiction avec la méthode du bagging
prediction_bagging = robot_predictor.faire_prediction_bagging()
resultat_bagging = robot_predictor.determiner_resultat(prediction_bagging)
print(f"Prédiction (Bagging) pour le prochain match entre {equipe_A} et {equipe_B} : {resultat_bagging}")

# Ajouter les nouvelles méthodes dans evaluer_methodes
resultats, methode_precise = robot_predictor.evaluer_methodes()

# Évaluer les méthodes et afficher les résultats
resultats, methode_precise = robot_predictor.evaluer_methodes()

# Afficher les résultats pour chaque méthode
print("\nRésultats pour chaque méthode :")
for methode, resultat in resultats.items():
    print(f"Prédiction ({methode}) : {resultat[1]} avec score prédit : {resultat[0]}")

# Afficher la méthode la plus précise
print(f"\nLa méthode la plus précise est : {methode_precise}")

# Obtenir et afficher les scores prédits selon chaque méthode de calcul
scores_predits = robot_predictor.obtenir_scores_predits()
print("\nScores prédits selon chaque méthode de calcul :")
for methode, score_predit in scores_predits.items():
    print(f"Prédiction ({methode}) : {score_predit}")
