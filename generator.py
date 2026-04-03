import time
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
import warnings
from joblib import Parallel, delayed
import os

warnings.filterwarnings("ignore")


def warmup():
    """ Funkcja rozgrzewająca, która wykonuje proste obliczenia, aby "rozgrzać" procesor i zoptymalizować wydajność podczas rzeczywistej symulacji. """
    Parallel(n_jobs=-1, batch_size="auto")(
    delayed(generate_test)(i, 1, 10, "easy", 100, 4, 0.25, 0.5) for i in range(10))
    return 


def cor_secure(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
  
def calculate_risk_probability(ability_diff):
    """ Oblicza prawdopodobieństwo podjęcia ryzyka strzału na podstawie różnicy między 
        trudnością pytania a umiejętnościami ucznia."""
    diff_clamped = np.clip(ability_diff, 0.0, 0.6) 
    prob = 1.0 - (diff_clamped * 1.5)
    return np.clip(prob, 0.1, 0.95)
 
def create_sections(num_sections):
    """ Tworzy listę działów z przypisaną etykietą trudności, wartością trudności i częstotliwością. """
    labels = ["easy", "medium", "medium-hard", "hard"]  # Odpowiadające etykiety: łatwy, średni, średnio-trudny, trudny
    section_ids = np.arange(0, num_sections)
    difficulty_label = np.random.choice(labels, size=num_sections)
    conlist = [difficulty_label == "easy", difficulty_label == "medium", difficulty_label == "medium-hard", difficulty_label == "hard"]
    checklist = [truncnorm.rvs((0.15 - 0.25)/0.05, (0.35 - 0.25)/0.05, loc=0.25, scale=0.05, size=num_sections).astype(np.float32),
                 truncnorm.rvs((0.35 - 0.45)/0.05, (0.55 - 0.45)/0.05, loc=0.45, scale=0.05, size=num_sections).astype(np.float32),
                 truncnorm.rvs((0.55 - 0.65)/0.05, (0.75 - 0.65)/0.05, loc=0.65, scale=0.05, size=num_sections).astype(np.float32),
                 truncnorm.rvs((0.75 - 0.85)/0.05, (0.90 - 0.85)/0.05, loc=0.85, scale=0.05, size=num_sections).astype(np.float32)]
    difficulty_values = np.select(conlist, checklist)

    return (difficulty_label, difficulty_values, section_ids)

def create_students(group_size, group_knowledge_level, sections):
    """Tworzy uczniów z ich stresem, poziomem wiedzy ogólnej oraz poziomem wiedzy z poszczególnych działów."""

    diversity = 0.15
    a_clip, b_clip = (0 - group_knowledge_level) / diversity, (1 - group_knowledge_level) / diversity
    knowledge_levels = truncnorm.rvs(a_clip, b_clip, loc=group_knowledge_level, scale=diversity, size=group_size).astype(np.float32)
    stress = truncnorm.rvs(0, 1, size=group_size).astype(np.float32)
    conlist =[stress < 0.3, (stress >= 0.3) & (stress < 0.65), stress >= 0.65]
    choicelist = [knowledge_levels, knowledge_levels + 0.1, knowledge_levels - 0.1]
    theta_values = np.select(conlist, choicelist)
    t = theta_values[:, np.newaxis]
    knowledge_by_section = truncnorm.rvs((0 - t)/0.15, (1 - t)/0.15, loc=t, scale=0.15, size=(group_size, len(sections[0]))).astype(np.float32)
    student_ids = np.arange(1, group_size + 1).reshape(-1, 1)
    knowledge_by_section_id = np.empty((group_size, len(sections[0]) + 1), dtype=np.float32)
    knowledge_by_section_id[:, 0] = student_ids.flatten()
    knowledge_by_section_id[:, 1:] = knowledge_by_section
    return theta_values, knowledge_by_section_id

def create_questions(sections):
    """Dzieli pytania na trzy poziomy trudności (łatwe, średnie, trudne) i przypisuje im trudność theta."""

    sections_values = sections[1]
    sections_ids = sections[2].astype(int)
    sections_labels = sections[0]
    values_matrix = np.repeat(sections_values, 3)
    sections_matrix = np.repeat(sections_ids, 3)
    section_labels_matrix = np.repeat(sections_labels, 3)
    difficulty_labels = np.tile(np.array(['easy', 'medium', 'hard']), len(sections[2]))
    difficulty = np.tile(np.array([-0.10, 0.0, 0.10]), len(sections[2])).astype(np.float32)
    question_theta = values_matrix + difficulty
    mask = (question_theta >= 0.0) & (question_theta <= 1.0)
    question_theta = question_theta[mask]
    sections_matrix = sections_matrix[mask]
    section_labels_matrix = section_labels_matrix[mask]
    difficulty_labels = difficulty_labels[mask]
    question_ids = np.arange(1, len(question_theta)+1)
    
    return (question_ids, sections_matrix, question_theta, difficulty_labels)

def create_test_structure(questions_pool):
    """ Tworzy strukturę testu, losowo wybierając pytania z puli na podstawie częstotliwości działów."""
    ids = questions_pool[0]
    section_ids = questions_pool[1]
    question_thetas = questions_pool[2]
    difficulty_labels = questions_pool[3]
    num_questions = random.randint(1, 101)
    size_l = len(ids)
    selected_test_questions = np.random.choice(size_l, size=num_questions, replace=True)
    test_question_ids = ids[selected_test_questions]
    test_section_ids = section_ids[selected_test_questions]
    test_question_thetas = question_thetas[selected_test_questions]
    test_difficulty_labels = difficulty_labels[selected_test_questions]
    
    return ( test_question_ids, test_section_ids, test_question_thetas, test_difficulty_labels )

def simulate_test_0(test_structure, knowledge_by_section, theta_values, guessing_prob):
    """ Symuluje przebieg testu dla każdego ucznia, uwzględniając ich wiedzę, poziom stresu, oraz prawdopodobieństwo podjęcia ryzyka. """

    len_students = knowledge_by_section.shape[0]
    

    id_of_sections = test_structure[1]
    q_theta_values = test_structure[2]
    len_questions = len(q_theta_values)
    knowladge_for_questions = knowledge_by_section[:, id_of_sections + 1]
    current_fatigue = (np.arange(len_questions) * 0.002).astype(np.float32)
    effective_knowledge = knowladge_for_questions - current_fatigue
    resolve_ability = effective_knowledge - q_theta_values
    conlist_guess = resolve_ability < 0
    scores = np.where(conlist_guess, np.random.binomial(1, guessing_prob, size=(len_students, len_questions)), 1)
    did_guess = np.where(conlist_guess, 1, 0)
    was_hit = np.where((conlist_guess) & (scores == 1), 1, np.where(conlist_guess, 0, -1))
    students_ids = np.repeat(knowledge_by_section[:, 0], len_questions)
    question_ids = np.tile(test_structure[0], len_students)
    student_thetas = np.repeat(theta_values, len_questions)
    
    return ( students_ids, question_ids, student_thetas, scores.flatten(), did_guess.flatten(), was_hit.flatten() )
        
def simulate_test_1(test_structure, knowledge_by_section, theta_values,  guessing_prob):
    """ Symuluje przebieg testu dla każdego ucznia, uwzględniając ich wiedzę, poziom stresu, oraz prawdopodobieństwo podjęcia ryzyka. """

    len_students = knowledge_by_section.shape[0]
    

    id_of_sections = test_structure[1]
    q_theta_values = test_structure[2]
    len_questions = len(q_theta_values)
    knowladge_for_questions = knowledge_by_section[:, id_of_sections + 1]
    current_fatigue = (np.arange(len_questions) * 0.002).astype(np.float32)  
    effective_knowledge = knowladge_for_questions - current_fatigue
    resolve_ability = effective_knowledge - q_theta_values
    conlist_guess = resolve_ability < 0
    risk_taking_prob = calculate_risk_probability(resolve_ability).astype(np.float32)
    random_values = np.random.rand(len_students, len_questions) 
    did_guess = np.where((conlist_guess) & (random_values < risk_taking_prob), 1, 0)
    scores = np.where(did_guess == 1, np.random.binomial(1, guessing_prob, size=(len_students, len_questions)) * 2 - 1, 
                np.where(resolve_ability >= 0, 1, 0))
    was_hit = np.where((did_guess == 1) & (scores == 1), 1, np.where(did_guess == 1, 0, -1))
    students_ids = np.repeat(knowledge_by_section[:, 0], len_questions)
    question_ids = np.tile(test_structure[0], len_students)
    student_thetas = np.repeat(theta_values, len_questions)
    
    return ( students_ids, question_ids, student_thetas, scores.flatten(), did_guess.flatten(), was_hit.flatten() )
    
def calculate_summary(results, num_questions):
    """ Podsumowuje wyniki testu dla każdego ucznia a następnie podaje ogólne metryki testu. """
    scores = results[3].reshape(-1, num_questions)
    did_guess = results[4].reshape(-1, num_questions)
    was_hit = results[5].reshape(-1, num_questions)
    student_ids = results[0].reshape(-1, num_questions)[:, 0]
    student_thetas = results[2].reshape(-1, num_questions)[:, 0]
    total_points = scores.sum(axis=1)
    guess_count = (did_guess == 1).sum(axis=1)
    hit_count = (was_hit == 1).sum(axis=1)
    final_score_pct = total_points / num_questions
    guess_percentage = guess_count / num_questions
    hit_rate = np.where(guess_count > 0, hit_count / guess_count, 0)
    grade = np.where(final_score_pct < 0.5, 2,
                np.where(final_score_pct < 0.6, 3,
                np.where(final_score_pct < 0.7, 3.5,
                np.where(final_score_pct < 0.8, 4,
                np.where(final_score_pct < 0.9, 4.5, 5)))))
    pass_rate = (grade > 2).sum() / len(student_ids)
    
    summary = (student_ids, student_thetas, total_points, guess_count, hit_count, final_score_pct, guess_percentage, hit_rate, grade)
    return summary, pass_rate

#@profile
def generate_test(test_id, test_type, num_sections_count, group_knowledge_label, group_size, num_options, guessing_prob, group_knowledge_level):
    """Główna funkcja generująca pojedyńczy test."""
    sections_data = create_sections(num_sections_count)
    

    theta_values, knowledge_by_section = create_students(
        group_size, 
        group_knowledge_level, 
        sections_data
    )
    
    questions_pool = create_questions(sections_data)
    
    test_structure = create_test_structure(
        questions_pool
    )
    if test_type == 0:
        results = simulate_test_0(
            test_structure, 
            knowledge_by_section, 
            theta_values, 
            guessing_prob
        )
    else:
        results = simulate_test_1(
            test_structure, 
            knowledge_by_section, 
            theta_values, 
            guessing_prob
        )
    num_questions = test_structure[0].shape[0]
    calculated_summary, pass_rate = calculate_summary(
        results, 
        num_questions
        )

    return ( test_id, num_questions, group_size, num_sections_count, num_options, group_knowledge_label, float(test_structure[2].mean()), float(test_structure[2].min()), float(test_structure[2].max()), (test_structure[3] == "easy").sum() / num_questions, (test_structure[3] == "medium").sum() / num_questions, (test_structure[3] == "hard").sum() / num_questions, cor_secure(calculated_summary[1], calculated_summary[5]), cor_secure(calculated_summary[5], calculated_summary[7]), pass_rate )
        

def simulate (num_tests, test_type):
    """ Funkcja do symulacji wielu testów i zbierania ich metryk. """

    start_time = time.time()

    num_sections_count = np.random.randint(1, 50, size=num_tests)
    labels = ["weak", "average", "advanced"]
    group_knowledge_label = np.random.choice(labels, size=num_tests)
    group_knowlage_checklist = [truncnorm.rvs((0.3 - 0.4)/0.02, (0.5 - 0.4)/0.05, loc=0.4, scale=0.05, size=num_tests).astype(np.float32),
                                truncnorm.rvs((0.5 - 0.6)/0.05, (0.7 - 0.6)/0.05, loc=0.6, scale=0.05, size=num_tests).astype(np.float32),
                                truncnorm.rvs((0.7 - 0.8)/0.05, (0.90 - 0.8)/0.05, loc=0.8, scale=0.05, size=num_tests).astype(np.float32)]
    group_knowledge_level = np.select([group_knowledge_label == "weak", group_knowledge_label == "average", group_knowledge_label == "advanced"], group_knowlage_checklist)

    group_size = np.random.randint(1, 200, size=num_tests)
    num_options = np.random.randint(2, 5, size=num_tests)
    guessing_prob = (1 / num_options).astype(np.float32) 
    
    simulations = Parallel(n_jobs=-1, batch_size="auto")(
    delayed(generate_test)(i, test_type, num_sections_count[i], group_knowledge_label[i], group_size[i], num_options[i], guessing_prob[i], group_knowledge_level[i]) for i in range(num_tests))

    column_names = [
        "test_id", "num_questions", "group_size", "num_sections", 
        "num_options", "group_knowledge_label", "mean_question_theta", 
        "min_question_theta", "max_question_theta", "pct_easy_questions", 
        "pct_medium_questions", "pct_hard_questions", "theta_vs_score_corr", 
        "score_vs_hits_corr", "pass_rate"
    ]

    data = pd.DataFrame(simulations, columns=column_names)
           
    end_time = time.time()
    duration = end_time - start_time

   

    return data, duration

if __name__ == "__main__":
    simulate(100, 0)