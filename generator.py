from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
from numba import njit
import warnings
warnings.filterwarnings("ignore")

def cor_secure(a, b):
    """ Zabezpiecza przed błędami w obliczaniu korelacji, 
        zwracając 0.0, gdy korelacja jest niezdefiniowana."""
    correlation = a.corr(b)
    if pd.isna(correlation):
        return 0.0
    else:
        return float(correlation)
    
@njit
def calculate_risk_probability_numba(ability_diff):
    """ Skompilowana wersja obliczania ryzyka. """
    # np.clip w Numbie działa błyskawicznie
    diff_clamped = np.clip(ability_diff, 0.0, 0.6) 
    prob = 1.0 - (diff_clamped * 1.5)
    return np.clip(prob, 0.1, 0.95)
def calculate_risk_probability(ability_diff):
    """ Oblicza prawdopodobieństwo podjęcia ryzyka strzału na podstawie różnicy między 
        trudnością pytania a umiejętnościami ucznia."""
    diff_clamped = np.clip(ability_diff, 0.0, 0.6) 
    prob = 1.0 - (diff_clamped * 1.5)
    return np.clip(prob, 0.1, 0.95)

def generate_knowledge(label):
    """ Generuje poziom wiedzy grupy na podstawie etykiety słownej. """
    if label == 'weak':
        return float(truncnorm.rvs((0.3 - 0.4)/0.02, (0.5 - 0.4)/0.05, loc=0.4, scale=0.05))
    elif label == 'average':
        return float(truncnorm.rvs((0.5 - 0.6)/0.05, (0.7 - 0.6)/0.05, loc=0.6, scale=0.05))
    elif label == 'advanced':
        return float(truncnorm.rvs((0.7 - 0.8)/0.05, (0.90 - 0.8)/0.05, loc=0.8, scale=0.05))
 
def create_sections(num_sections):
    """ Tworzy listę działów z przypisaną etykietą trudności, wartością trudności i częstotliwością. """
    labels = ["easy", "medium", "medium-hard", "hard"]  # Odpowiadające etykiety: łatwy, średni, średnio-trudny, trudny
    section_ids = np.arange(0, num_sections)
    difficulty_label = np.random.choice(labels, size=num_sections)
    conlist = [difficulty_label == "easy", difficulty_label == "medium", difficulty_label == "medium-hard", difficulty_label == "hard"]
    checklist = [truncnorm.rvs((0.15 - 0.25)/0.05, (0.35 - 0.25)/0.05, loc=0.25, scale=0.05, size=num_sections),
                 truncnorm.rvs((0.35 - 0.45)/0.05, (0.55 - 0.45)/0.05, loc=0.45, scale=0.05, size=num_sections),
                 truncnorm.rvs((0.55 - 0.65)/0.05, (0.75 - 0.65)/0.05, loc=0.65, scale=0.05, size=num_sections),
                 truncnorm.rvs((0.75 - 0.85)/0.05, (0.90 - 0.85)/0.05, loc=0.85, scale=0.05, size=num_sections)]
    difficulty_values = np.select(conlist, checklist)
    sections = list(zip(difficulty_label, difficulty_values, section_ids))  
    return sections

def create_students(group_size, group_knowledge_level, sections):
    """Tworzy uczniów z ich stresem, poziomem wiedzy ogólnej oraz poziomem wiedzy z poszczególnych działów."""
    diversity = 0.15
    a_clip, b_clip = (0 - group_knowledge_level) / diversity, (1 - group_knowledge_level) / diversity
    knowledge_levels = truncnorm.rvs(a_clip, b_clip, loc=group_knowledge_level, scale=diversity, size=group_size)
    stress = truncnorm.rvs(0, 1, size=group_size)
    conlist =[stress < 0.3, (stress >= 0.3) & (stress < 0.65), stress >= 0.65]
    choicelist = [knowledge_levels, knowledge_levels + 0.1, knowledge_levels - 0.1]
    theta_values = np.select(conlist, choicelist)
    t = theta_values[:, np.newaxis]
    knowledge_by_section = truncnorm.rvs((0 - t)/0.15, (1 - t)/0.15, loc=t, scale=0.15, size=(group_size, len(sections)))
    student_ids = np.arange(1, group_size + 1).reshape(-1, 1)
    knowledge_by_section = np.hstack((student_ids, knowledge_by_section))
    return theta_values, knowledge_by_section

def create_questions(sections):
    """Dzieli pytania na trzy poziomy trudności (łatwe, średnie, trudne) i przypisuje im trudność theta."""

    questions_pool = {}
    q_id = 0
    for section_name, diff, s_id in sections:
        if diff >= 0.10:
            q_id += 1
            questions_pool[q_id] = [s_id, section_name, diff - 0.10, 'easy']
        q_id += 1
        questions_pool[q_id] = [s_id, section_name, diff, 'medium']
        if diff <= 0.90:
            q_id += 1
            questions_pool[q_id] = [s_id, section_name, diff + 0.10, 'hard']
    return questions_pool

def create_test_structure(questions_pool):
    """ Tworzy strukturę testu, losowo wybierając pytania z puli na podstawie częstotliwości działów."""
    pool_keys = list(questions_pool.keys())  
    num_questions = random.randint(1, 100)
    selected_test_questions = tuple(random.choices(pool_keys, k=num_questions))

    test_structure = []
    test_table_data = []
    temp_id = 0
    for q_key in selected_test_questions:
        temp_id += 1
        test_structure.append([temp_id,questions_pool[q_key][0], questions_pool[q_key][2]])
        test_table_data.append({
            "question_id": temp_id,
            "section": questions_pool[q_key][1],
            "question_theta": questions_pool[q_key][2],
            "difficulty_level": questions_pool[q_key][3]
        })
    return np.array(test_structure), pd.DataFrame(test_table_data)

def simulate_test(test_structure, knowledge_by_section, theta_values, test_type, guessing_prob):
    """ Symuluje przebieg testu dla każdego ucznia, uwzględniając ich wiedzę, poziom stresu, oraz prawdopodobieństwo podjęcia ryzyka. """

    len_students = knowledge_by_section.shape[0]
    len_questions = len(test_structure)

    id_of_sections = test_structure[:,1].astype(int)
    q_theta_values = test_structure[:,2].astype(float)
    knowladge_for_questions = knowledge_by_section[:, id_of_sections + 1]
    current_fatigue = np.arange(len_questions) * 0.002
    effective_knowledge = knowladge_for_questions - current_fatigue
    resolve_ability = effective_knowledge - q_theta_values
    conlist_guess = resolve_ability < 0
    if test_type == 0:
        scores = np.where(conlist_guess, np.random.binomial(1, guessing_prob, size=(len_students, len_questions)), 1)
        did_guess = np.where(conlist_guess, "yes", "no")
        was_hit = np.where((conlist_guess) & (scores == 1), "yes", np.where(conlist_guess, "no", "-"))
    else:
        risk_taking_prob = calculate_risk_probability(resolve_ability)
        random_values = np.random.rand(len_students, len_questions)
        did_guess = np.where((conlist_guess) & (random_values < risk_taking_prob), "yes", "no")
        scores = np.where(did_guess == "yes", 
                  np.random.binomial(1, guessing_prob, size=(len_students, len_questions)) * 2 - 1, 
                  np.where(resolve_ability >= 0, 1, 0))
        was_hit = np.where((did_guess == "yes") & (scores == 1), "yes", np.where(did_guess == "yes", "no", "-"))

    students_ids = np.repeat(knowledge_by_section[:, 0], len_questions)
    question_ids = np.tile(test_structure[:, 0], len_students)
    student_thetas = np.repeat(theta_values, len_questions)
    results_df = pd.DataFrame({
        "student_id": students_ids,
        "question_id": question_ids,
        "student_theta": student_thetas,
        "score": scores.flatten(),
        "did_guess": did_guess.flatten(),
        "was_hit": was_hit.flatten()
    })
    
    return results_df


def calculate_summary(results_df, num_questions, score_thresholds):
    """ Podsumowuje wyniki testu dla każdego ucznia a następnie podaje ogólne metryki testu. """
    summary = (
        results_df
        .groupby(["student_id", "student_theta"])
        .agg(
            total_points=("score", "sum"),
            guess_count=("did_guess", lambda x: (x == "yes").sum()),
            hit_count=("was_hit", lambda x: (x == "yes").sum()),
        )
        .reset_index()
    )

    summary["final_score_pct"] = summary["total_points"] / num_questions
    summary["guess_percentage"] = summary["guess_count"] / num_questions
    summary["hit_rate"] = summary.apply(
        lambda row: row["hit_count"] / row["guess_count"] if row["guess_count"] > 0 else 0, axis=1
    )

    summary["grade"] = summary["final_score_pct"].apply(
        lambda x: 2   if x < score_thresholds[0] else
                  3   if x < score_thresholds[1] else       
                  3.5 if x < score_thresholds[2] else  #ponownie wektoryzacja z NUMPY 
                  4   if x < score_thresholds[3] else
                  4.5 if x < score_thresholds[4] else
                  5 )

    pass_count = (summary["grade"] > 2).sum()
    pass_rate = pass_count / len(summary) if len(summary) > 0 else 0

    return summary, pass_rate

#@profile
def generate_test(test_id, test_type):
    """Główna funkcja generująca pojedyńczy test."""
    
    score_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    num_sections_count = random.randint(1, 50)
    sections_data = create_sections(num_sections_count)
    group_knowledge_label = random.choice(["weak", "average", "advanced"])
    group_knowledge_level = generate_knowledge(group_knowledge_label)
    group_size = random.randint(1, 200)
    num_options = random.randint(2, 5)
    guessing_prob = 1 / num_options

    theta_values, knowledge_by_section = create_students(
        group_size, 
        group_knowledge_level, 
        sections_data
    )
    
    questions_pool = create_questions(sections_data)
    
    test_structure, test_df = create_test_structure(
        questions_pool
    )
  
    results_df = simulate_test(
        test_structure, 
        knowledge_by_section, 
        theta_values, 
        test_type, 
        guessing_prob
    )

    summary_df, pass_rate = calculate_summary(
        results_df, 
        len(test_structure), 
        score_thresholds
    )

    return {
        "test_id": test_id,
        "num_questions": len(test_structure),
        "group_size": group_size,
        "num_sections": num_sections_count,
        "num_options": num_options,
        "group_knowledge_label": group_knowledge_label,
        "mean_question_theta": float(test_df["question_theta"].mean()),
        "min_question_theta": float(test_df["question_theta"].min()),
        "max_question_theta": float(test_df["question_theta"].max()),
        "pct_easy_questions": (test_df["difficulty_level"] == "easy").sum() / len(test_structure),
        "pct_medium_questions": (test_df["difficulty_level"] == "medium").sum() / len(test_structure),
        "pct_hard_questions": (test_df["difficulty_level"] == "hard").sum() / len(test_structure),
        "theta_vs_score_corr": cor_secure(summary_df["student_theta"], summary_df["final_score_pct"]),
        "score_vs_hits_corr": cor_secure(summary_df["final_score_pct"], summary_df["hit_rate"]),
        "pass_rate": pass_rate
    }

