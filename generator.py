from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import random
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

    return { 
        "labels": difficulty_label, 
        "values": difficulty_values, 
        "ids": section_ids 
    }

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
    knowledge_by_section = truncnorm.rvs((0 - t)/0.15, (1 - t)/0.15, loc=t, scale=0.15, size=(group_size, len(sections["ids"])))
    student_ids = np.arange(1, group_size + 1).reshape(-1, 1)
    knowledge_by_section = np.hstack((student_ids, knowledge_by_section))
    return theta_values, knowledge_by_section

def create_questions(sections):
    """Dzieli pytania na trzy poziomy trudności (łatwe, średnie, trudne) i przypisuje im trudność theta."""

    sections_values = sections["values"]
    sections_ids = sections["ids"].astype(int)
    sections_labels = sections["labels"]
    values_matrix = np.repeat(sections_values, 3)
    sections_matrix = np.repeat(sections_ids, 3)
    section_labels_matrix = np.repeat(sections_labels, 3)
    difficulty_labels = np.tile(np.array(['easy', 'medium', 'hard']), len(sections["ids"]))
    difficulty = np.tile(np.array([-0.10, 0.0, 0.10]), len(sections["ids"]))
    question_theta = values_matrix + difficulty
    mask = (question_theta >= 0.0) & (question_theta <= 1.0)
    question_theta = question_theta[mask]
    sections_matrix = sections_matrix[mask]
    section_labels_matrix = section_labels_matrix[mask]
    difficulty_labels = difficulty_labels[mask]
    question_ids = np.arange(1, len(question_theta)+1)
    
    return {
        "ids": question_ids,
        "section_ids": sections_matrix,
        "thetas": question_theta,
        "labels": difficulty_labels
    }

def create_test_structure(questions_pool):
    """ Tworzy strukturę testu, losowo wybierając pytania z puli na podstawie częstotliwości działów."""
    ids = questions_pool["ids"]
    section_ids = questions_pool["section_ids"]
    question_thetas = questions_pool["thetas"]
    difficulty_labels = questions_pool["labels"]
    num_questions = random.randint(1, 101)
    size_l = len(ids)
    selected_test_questions = np.random.choice(size_l, size=num_questions, replace=True)
    test_question_ids = ids[selected_test_questions]
    test_section_ids = section_ids[selected_test_questions]
    test_question_thetas = question_thetas[selected_test_questions]
    test_difficulty_labels = difficulty_labels[selected_test_questions]
    
    return {
        "question_id": test_question_ids,
        "section": test_section_ids,
        "question_theta": test_question_thetas,
        "difficulty_level": test_difficulty_labels
    }


def simulate_test_0(test_structure, knowledge_by_section, theta_values, guessing_prob):
    """ Symuluje przebieg testu dla każdego ucznia, uwzględniając ich wiedzę, poziom stresu, oraz prawdopodobieństwo podjęcia ryzyka. """

    len_students = knowledge_by_section.shape[0]
    

    id_of_sections = test_structure["section"]
    q_theta_values = test_structure["question_theta"]
    len_questions = len(q_theta_values)
    knowladge_for_questions = knowledge_by_section[:, id_of_sections + 1]
    current_fatigue = np.arange(len_questions) * 0.002
    effective_knowledge = knowladge_for_questions - current_fatigue
    resolve_ability = effective_knowledge - q_theta_values
    conlist_guess = resolve_ability < 0
    scores = np.where(conlist_guess, np.random.binomial(1, guessing_prob, size=(len_students, len_questions)), 1)
    did_guess = np.where(conlist_guess, 1, 0)
    was_hit = np.where((conlist_guess) & (scores == 1), 1, np.where(conlist_guess, 0, -1))
    students_ids = np.repeat(knowledge_by_section[:, 0], len_questions)
    question_ids = np.tile(test_structure["question_id"], len_students)
    student_thetas = np.repeat(theta_values, len_questions)
    
    return {
        "student_id": students_ids,
        "question_id": question_ids,
        "student_theta": student_thetas,
        "score": scores.flatten(),
        "did_guess": did_guess.flatten(),
        "was_hit": was_hit.flatten()
    }

def simulate_test_1(test_structure, knowledge_by_section, theta_values,  guessing_prob):
    """ Symuluje przebieg testu dla każdego ucznia, uwzględniając ich wiedzę, poziom stresu, oraz prawdopodobieństwo podjęcia ryzyka. """

    len_students = knowledge_by_section.shape[0]
    

    id_of_sections = test_structure["section"]
    q_theta_values = test_structure["question_theta"]
    len_questions = len(q_theta_values)
    knowladge_for_questions = knowledge_by_section[:, id_of_sections + 1]
    current_fatigue = np.arange(len_questions) * 0.002
    effective_knowledge = knowladge_for_questions - current_fatigue
    resolve_ability = effective_knowledge - q_theta_values
    conlist_guess = resolve_ability < 0
    risk_taking_prob = calculate_risk_probability(resolve_ability)
    random_values = np.random.rand(len_students, len_questions) 
    did_guess = np.where((conlist_guess) & (random_values < risk_taking_prob), 1, 0)
    scores = np.where(did_guess == 1, np.random.binomial(1, guessing_prob, size=(len_students, len_questions)) * 2 - 1, 
                np.where(resolve_ability >= 0, 1, 0))
    was_hit = np.where((did_guess == 1) & (scores == 1), 1, np.where(did_guess == 1, 0, -1))
    students_ids = np.repeat(knowledge_by_section[:, 0], len_questions)
    question_ids = np.tile(test_structure["question_id"], len_students)
    student_thetas = np.repeat(theta_values, len_questions)
    
    return {
        "student_id": students_ids,
        "question_id": question_ids,
        "student_theta": student_thetas,
        "score": scores.flatten(),
        "did_guess": did_guess.flatten(),
        "was_hit": was_hit.flatten()
    }
    


def calculate_summary(results, num_questions):
    """ Podsumowuje wyniki testu dla każdego ucznia a następnie podaje ogólne metryki testu. """
    scores = results["score"].reshape(-1, num_questions)
    did_guess = results["did_guess"].reshape(-1, num_questions)
    was_hit = results["was_hit"].reshape(-1, num_questions)
    student_ids = results["student_id"].reshape(-1, num_questions)[:, 0]
    student_thetas = results["student_theta"].reshape(-1, num_questions)[:, 0]
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
    
    summary = { "student_id": student_ids ,
               "student_theta": student_thetas,
               "total_points": total_points,
               "guess_count": guess_count,
               "hit_count": hit_count,
               "final_score_pct": final_score_pct,
               "guess_percentage": guess_percentage,
               "hit_rate": hit_rate,
               "grade": grade
    }
    return summary, pass_rate
    
    

#@profile
def generate_test(test_id, test_type):
    """Główna funkcja generująca pojedyńczy test."""
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
    num_questions = test_structure["question_id"].shape[0]
    summary, pass_rate = calculate_summary(
        results, 
        num_questions
        )

    return {
        "test_id": test_id,
        "num_questions": num_questions,
        "group_size": group_size,
        "num_sections": num_sections_count,
        "num_options": num_options,
        "group_knowledge_label": group_knowledge_label,
        "mean_question_theta": float(test_structure["question_theta"].mean()),
        "min_question_theta": float(test_structure["question_theta"].min()),
        "max_question_theta": float(test_structure["question_theta"].max()),
        "pct_easy_questions": (test_structure["difficulty_level"] == "easy").sum() / num_questions,
        "pct_medium_questions": (test_structure["difficulty_level"] == "medium").sum() / num_questions,
        "pct_hard_questions": (test_structure["difficulty_level"] == "hard").sum() / num_questions,
        "theta_vs_score_corr": cor_secure(pd.Series(summary["student_theta"]), pd.Series(summary["final_score_pct"])),
        "score_vs_hits_corr": cor_secure(pd.Series(summary["final_score_pct"]), pd.Series(summary["hit_rate"])),
        "pass_rate": pass_rate
    }

