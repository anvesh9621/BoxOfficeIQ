"""
DistilBERT fine-tuning script for movie review sentiment analysis.
Trains on ml/data/raw/sample_reviews.csv or generates 500 synthetic reviews.
Saves model to ml/models/sentiment_model/ and tokenizer to ml/models/sentiment_tokenizer/.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "ml" / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
MODEL_DIR = BASE_DIR / "ml" / "models" / "sentiment_model"
TOKENIZER_DIR = BASE_DIR / "ml" / "models" / "sentiment_tokenizer"
REVIEWS_CSV = RAW_DIR / "sample_reviews.csv"
METRICS_JSON = PROCESSED_DIR / "sentiment_metrics.json"
SCORES_CSV = PROCESSED_DIR / "review_scores.csv"

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, TOKENIZER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic review data (used if CSV not found)
# ---------------------------------------------------------------------------
POSITIVE_REVIEWS = [
    "This film was absolutely brilliant. The storyline kept me on the edge of my seat throughout. Outstanding performances from every cast member. A true cinematic masterpiece.",
    "Fantastic performances and a gripping story that never let go. The director's vision is clear and compelling. I was moved to tears by the third act. Highly recommend to everyone.",
    "A stunning piece of cinema that blends drama and emotion seamlessly. The cinematography is breathtaking and the score perfectly complements each scene. One of the best films of the year.",
    "An extraordinary movie experience from start to finish. The character development is rich and believable. The pacing is perfect and never drags. I would watch this again and again.",
    "Incredible storytelling with nuanced characters I genuinely cared about. The dialogue feels natural and the performances are top-notch. A film that stays with you long after the credits roll.",
    "Simply one of the best movies I have seen in years. Every scene is crafted with care and intention. The ensemble cast delivers brilliantly across the board. A must-watch.",
    "A breathtaking cinematic journey filled with emotion and suspense. The director handles the material with great sensitivity. The ending left me speechless. Absolutely loved it.",
    "This movie exceeded every expectation I had. The script is sharp and witty while also being deeply moving. The lead performance is award-worthy. Pure cinematic gold.",
    "Riveting from the very first frame to the very last. The tension builds masterfully and never lets up. The chemistry between the leads is electric. One of the year's best.",
    "A powerhouse film that delivers on every level. Great direction, superb acting, and a story with real heart. The production design is stunning. This is what movies should be.",
    "Beautifully crafted with a story that resonates deeply. The performances are uniformly excellent and the direction is assured. I found myself completely absorbed from start to finish.",
    "A triumph of filmmaking craft and emotional storytelling. The script balances humor and heartbreak perfectly. The cinematography is gorgeous and inventive. A film not to be missed.",
    "Genuinely moving and endlessly entertaining. The performances are layered and authentic. The music score elevates every emotional beat. A wonderful film that deserves wide appreciation.",
    "This is cinema at its finest — intelligent, emotional, and visually spectacular. Every scene feels purposeful and earned. The climax is enormously satisfying. An instant classic.",
    "A magnificent film that fires on all cylinders. The story is original, the acting is phenomenal, and the direction is visionary. I cannot recommend this movie highly enough.",
    "Exceptional in every regard. The world-building is immersive, the characters feel real, and the pacing is flawless. A rare film that entertains and enlightens simultaneously.",
    "One of those rare films that transcends genre expectations. The writing is sharp and insightful. The lead actor gives a career-best performance. Absolutely deserving of all the praise.",
    "A deeply satisfying watch from beginning to end. The story unfolds with confidence and precision. The emotional beats land exactly as intended. I left the cinema feeling uplifted.",
    "Flawless execution of a compelling premise. The ensemble cast is outstanding and clearly invested in the material. The direction keeps things brisk and exciting. Essential viewing.",
    "This film captivated me completely. The narrative is layered and rewards close attention. The performances are understated but powerful. A gem of modern cinema.",
    "A joy from start to finish. The script is clever, the performances delightful, and the direction sure-footed. Exactly the kind of film the industry needs more of. Bravo.",
    "Moving, funny, and utterly compelling. The film manages to be entertaining and emotionally resonant without feeling manipulative. A wonderful achievement in modern storytelling.",
    "Visually stunning and emotionally gripping. The story takes unexpected turns that feel entirely earned. The final act is a masterclass in building and releasing tension. Superb.",
    "A film that reminded me why I love movies. The performances are raw and authentic. The direction is inventive without being showy. A deeply human story told with great skill.",
    "Smart, funny, and heartfelt all at once. The writing is fresh and unpredictable. The performances bring every character to vivid life. A fantastic film in every sense.",
    "An absolute delight that exceeded all my hopes. The story moves swiftly but never feels rushed. The chemistry between the leads is irresistible. One of my favorites this year.",
    "A film of rare ambition and rarer execution. The themes are weighty but handled with a light touch. The performances anchor the film with real humanity. Wonderful work.",
    "Stunningly good on every dimension. The cinematography, the score, the performances — all excellent. The story has real emotional depth. A film I will think about for years.",
    "What a remarkable piece of filmmaking. The director creates a fully realized world that you never want to leave. The performances are uniformly strong. A must-see.",
    "Gripping, moving, and visually arresting. This film does everything right. The story is original, the characters are vivid, and the ending is deeply satisfying. Highly recommended.",
    "A complete cinematic triumph that fires on every cylinder. Brilliant performances backed by an exceptional script. The pacing is immaculate and the finale delivers. Outstanding.",
    "One of the most memorable cinema experiences I have had in a long time. Rich characters, an absorbing plot, and production values through the roof. Simply unmissable.",
    "Warm, witty, and wonderfully acted. The script sparkles with sharp dialogue. The cast have palpable chemistry. A crowd-pleasing film that also has genuine depth.",
    "An emotionally devastating yet ultimately hopeful film. The writing walks the tonal tightrope perfectly. The lead gives an extraordinary performance. A landmark film.",
    "Sharp, smart cinema that trusts its audience completely. The story never condescends and always surprises. Exceptional performances all round. One of the finest films in recent memory.",
    "This film has everything — suspense, heart, humour, and spectacle. The director juggles the tones masterfully. A joy to watch from opening to close. Brilliantly done.",
    "Superb filmmaking with soul. The script has real insight into human nature. Every performance feels lived-in and truthful. A film that moves you while dazzling you. Unmissable.",
    "A beautifully observed film full of small, true moments. The characters feel like real people you have known all your life. Touching, funny, and deeply humane. Very highly recommended.",
    "Breathtakingly original and deeply moving. The director's control of tone and pacing is extraordinary. The cast deliver career-defining work. A film that will endure.",
    "Magnificent. That is the only word for it. Every frame is composed with care; every line of dialogue earns its place. A profound and exhilarating work of art.",
    "A film that left me completely floored. Innovative storytelling meets powerhouse acting and extraordinary direction. I cannot stop thinking about it. See it immediately.",
    "Joyful, surprising, and genuinely moving. The script avoids every cliché and finds fresh truth in its story. The performances are a delight. A special film.",
    "A stunning debut that announces a major new filmmaking voice. The vision is completely assured. The cast deliver wholeheartedly. An extraordinary achievement.",
    "Rich with detail and brimming with life. The world of the film feels utterly real. The ensemble is pitch-perfect. A satisfying, substantial piece of cinema.",
    "An emotionally generous film that never cheats. The story earns every tear and every laugh. The direction is elegant and invisible. A film to treasure.",
    "Spellbinding from first frame to last. The script is a marvel of construction. The lead performance is of the highest order. Essential cinema.",
    "A love letter to storytelling itself. Inventive, heartfelt, and technically dazzling. The performers give everything. A film full of pure, undiluted cinema magic.",
    "Genuinely gripping thriller with intelligence and heart. The twists are earned and the tension relentless. The cast elevate the material superbly. A very fine film.",
    "Every performance, every shot, every note of the score is perfectly calibrated. The film builds to a devastating and cathartic climax. A towering work.",
    "This is exactly the kind of bold, uncompromising cinema we need. Brave storytelling, extraordinary acting, visionary direction. I was shaken to my core. Unmissable.",
    "An absolute pleasure from start to finish. The film sparkles with wit and warmth. The ensemble is a delight. Feel-good cinema that also says something meaningful.",
    "Heartfelt and hugely entertaining. The story is refreshingly original. The performances carry enormous emotional weight. A film that earns its standing ovation.",
    "Sharp, clever, and unexpectedly touching. The script has real literary quality. The cast make it sing. A film to savour and return to.",
    "A glorious spectacle with genuine emotional depth. The action set-pieces are breathtaking but it is the character work that really lingers. Wonderful filmmaking.",
    "An unmissable work that sets a new benchmark. Technically flawless and emotionally devastating. The director has outdone themselves. A film for the ages.",
    "Perfectly judged in every respect. The story is deceptively simple but resonates powerfully. The performances are quietly superb. A gem of contemporary cinema.",
    "Riveting cinema that holds you completely. The script is tight and intelligent. Every actor is fully committed. A hugely satisfying and rewarding experience.",
    "One of those films that reminds you of the power of the medium. Poetic, profound, and profoundly moving. The director achieves something truly special here.",
    "A wonderfully surprising film that defies easy categorisation. The tone shifts are handled brilliantly. The cast is magnificent. An original and deeply rewarding work.",
    "Full of invention and genuine feeling. The story goes places you do not expect and earns every moment. The ensemble cast deliver brilliantly. A joy.",
    "Electrifying cinema that crackles with energy and ideas. The script is as smart as it is exciting. The performances match the material stride for stride. Exceptional.",
    "An outstanding film on every level. The writing is literate and exciting. The direction is both muscular and precise. The cast is extraordinary. See it.",
    "A film of immense beauty and power. The images linger in the mind long after the lights go up. The performances are indelible. Major cinema.",
    "Extraordinary in ambition and execution. The story grapples with big themes without losing sight of its human heart. The cast is magnificent. A defining film.",
    "Lush, engrossing, and emotionally resonant. The period detail is exquisite. The performances are rich and layered. A sumptuous and satisfying experience.",
    "Funny and fierce and full of life. The script has tremendous energy. The lead performance is a force of nature. A hugely enjoyable and nourishing film.",
    "A deeply assured piece of work. The direction is confident, the script precise, the performances committed. A film that grows on you with every passing day.",
    "Uplifting and thought-provoking in equal measure. The film has the courage of its convictions. The cast deliver with conviction. A meaningful and powerful work.",
    "A masterwork of genre filmmaking. The director understands every convention and either subverts or elevates them. The cast are uniformly excellent. Essential.",
    "One of the great screen performances at the centre of an already excellent film. The script is tight and the direction inspired. A complete triumph.",
    "Genuinely original and deeply felt. The film moves with a confidence that is never arrogant. The ensemble cast make it feel completely alive. A delight.",
    "Gorgeous and gripping in equal measure. The director composes images of real beauty. The story has tremendous emotional force. A film of rare quality.",
    "A perfectly pitched comedy drama that makes you laugh and cry in equal measure. The writing is superb. The cast are at the top of their game. Wonderful.",
    "An immensely satisfying and grown-up piece of cinema. The themes are handled with real intelligence. The performances are outstanding. A very special film.",
    "Compelling, surprising, and ultimately very moving. The story earns every emotion it generates. The cast is flawless. A film that will stay with you.",
    "A remarkable achievement in popular cinema. The director delivers spectacle and substance simultaneously. The cast rise magnificently to the material. Essential viewing.",
    "Rich, humane, and deeply enjoyable. The script is wise and witty. The performances are irresistible. A film to cherish.",
    "Technically brilliant and emotionally honest. The film trusts its audience completely. The ensemble is pitch-perfect. A very fine work indeed.",
    "Exhilarating and deeply moving. The film manages to be a complete crowd-pleaser while also having something profound to say. The cast deliver magnificently. Exceptional.",
    "A triumph of popular filmmaking. The director's command of tone and pacing is complete. The cast elevate an already excellent script. Unmissable entertainment.",
    "Quiet, beautiful, and profoundly moving. The director lets the story breathe and find its own rhythm. The performances have genuine depth. A treasurable film.",
    "A film of wit, warmth, and real wisdom. The script is a pleasure in every line. The cast make it sparkle. Pure, undiluted pleasure from start to finish.",
    "Dazzling and deeply human. The film operates on multiple levels simultaneously. The performances are extraordinary. A film of real lasting importance.",
    "Superb in every sense. The story is enthralling, the direction inspired, the performances magnificent. I was gripped from the first moment to the last. Highly recommended.",
    "A film that earns its emotions completely. No cheap tricks, no easy resolutions. Just honest, beautifully crafted storytelling. An exceptional work.",
    "Inventive, funny, and genuinely touching. The script has real personality. The director handles the material with great skill. A hugely enjoyable film.",
    "Sharp, smart, and full of genuine feeling. The film has real ambition and achieves it. The cast deliver with great skill. A film to be proud of.",
    "One of the most purely enjoyable films in years. The script is witty and inventive. The ensemble cast is superb. A film that makes you glad to be alive.",
    "Remarkable storytelling at every level. The film builds with great confidence towards a cathartic and satisfying climax. The performances are exceptional. A gem.",
    "Beautifully written, impeccably directed, brilliantly acted. The film fires on every cylinder from start to finish. A complete and deeply satisfying work.",
    "Thoughtful, funny, and unexpectedly moving. The script avoids all the obvious pitfalls. The cast deliver with evident joy. A film that works on every level.",
    "A film of immense skill and genuine heart. The director achieves exactly what the story demands. The cast make it feel completely real. Very highly recommended.",
    "An outstanding piece of cinema that works both as entertainment and as art. The script is excellent, the direction inspired. The performances make it sing. Essential.",
    "One of the most satisfying films of recent years. Rich in character and incident. The performances carry the film magnificently. A wholly rewarding experience.",
    "Bold, original, and deeply affecting. The film takes real risks and they all pay off. The cast is extraordinary. A major and important work of cinema.",
    "Beautifully shot and superbly performed. The film has genuine emotional power and uses it wisely. The story is absorbing and credible. A wonderful piece of work.",
    "Smart, gripping, and deeply moving. The film operates at the highest level throughout. The cast delivers uniformly excellent work. A film of real consequence.",
    "A fresh, funny, and deeply human film. The script sparkles with wit and insight. The performances are uniformly terrific. One of the year's most delightful surprises.",
    "An extraordinary film that justifies every bit of its considerable ambition. The storytelling is masterly. The performances are superb. An essential cinematic experience.",
    "Complete, satisfying, and genuinely moving. The film does not put a foot wrong from first frame to last. The cast make it feel completely alive. A joy.",
    "A film full of wonderful moments that add up to something truly special. Funny, moving, and beautifully crafted. The ensemble cast is a delight. Essential viewing.",
    "An absolute triumph. Funny, heartfelt, and technically masterful. The cast deliver with great warmth and skill. A film to watch, re-watch, and treasure.",
    "Staggeringly good cinema that fires on every level. The script is as smart as it is entertaining. The performances are indelible. A film of real greatness.",
    "A wonderful, full-hearted film that earns every emotion it generates. The storytelling is generous and intelligent. The cast is magnificent. Very highly recommended.",
    "Every element of this film is working at the highest level. The direction is visionary, the script superb, the performances extraordinary. A film for the ages.",
    "Entertaining, thought-provoking, and deeply moving by turns. The film has total command of its material. The cast is flawless. Essential, unmissable cinema.",
    "A breathtaking film that left me speechless. The performances are beyond compare. The story is told with extraordinary grace and skill. An absolute masterpiece.",
    "Joyous, funny, and profoundly moving. The film is generous in spirit and masterful in execution. The ensemble is delightful. A film to treasure and revisit.",
    "A film that takes your breath away. The storytelling is superb. The performances are of the very highest calibre. The images linger long. An unforgettable work.",
    "Completely absorbing and deeply satisfying. The film is never less than brilliant. The cast make it feel utterly real and alive. A milestone in modern cinema.",
    "A magnificent film of great power and subtlety. The direction is assured, the script outstanding, the performances extraordinary. A film of rare quality and importance.",
    "Superb, moving, and hugely entertaining. The director balances the film's many tones with great skill. The cast delivers wholeheartedly. Essential cinema.",
    "Funny, touching, and visually inventive. The film trusts its audience and rewards them handsomely. The cast is delightful. A wonderful piece of popular cinema.",
    "An almost impossibly good film. It delivers on every level and then some. The performances are extraordinary. The direction is visionary. Essential and unmissable.",
    "A film of rare and extraordinary quality. The story resonates deeply. The performances are of the very highest order. The direction is masterly. A must-see.",
    "Brilliant storytelling allied to brilliant filmmaking and brilliant acting. The film is a complete artistic and popular success. Hugely recommended. Essential viewing.",
    "One of the finest films ever made in this genre. The script is a marvel. The cast is extraordinary. The direction is visionary. Essential cinema of the highest order.",
    "Rich, satisfying, and deeply human. The film delivers everything it promises and much more. The performances are magnificent. Unmissable. Very highly recommended.",
    "Perfect cinema. The script is a gem. The performances are extraordinary. The direction is inspired. The ending is devastating and right. An absolute masterpiece.",
    "A film that makes you proud of what cinema can achieve. Funny, moving, profound, and visually spectacular. The ensemble is extraordinary. A landmark work.",
    "Completely exceptional. Rich storytelling, extraordinary performances, masterly direction. I was mesmerised from first to last. Essential cinema of the very highest order.",
    "One of the great films. Period. The script is remarkable. The performances are beyond praise. The direction is inspired. An essential and enduring work of art.",
    "Staggeringly assured and deeply affecting. The film is a complete artistic success. The performances are of the highest order. A film of rare and lasting power.",
    "A film to restore your faith in cinema. Original, daring, beautifully crafted, and profoundly moving. The cast delivers astonishing work. An absolute triumph.",
    "Simply one of the best films ever made. Rich in humanity and storytelling power. The performances are peerless. The direction flawless. An unforgettable masterpiece.",
    "A film that stands beside the very greatest works of cinema. Extraordinary in every dimension. Essential viewing for anyone who loves what film can achieve.",
    "The best film I have seen in years. Every frame is worth looking at. Every performance is perfect. Every line of dialogue earns its place. Essential cinema.",
    "An astonishing work of art. The film moves, dazzles, and illuminates in equal measure. The cast is peerless. The director has created something enduring. A masterpiece.",
    "Overwhelming in its power and beauty. The film achieves the highest ambitions of its storytelling. The performances are extraordinary. A work of rare and lasting greatness.",
    "Beyond exceptional. The film operates at a level that few works of popular art ever reach. Brilliant in every respect. Essential and unmissable. A masterwork.",
]

NEGATIVE_REVIEWS = [
    "Terrible waste of time and money. The plot made absolutely no sense from beginning to end. The acting was wooden and unconvincing. I deeply regret watching this.",
    "Poorly written and painfully boring. The story goes nowhere for nearly two hours. The characters are flat and completely uninteresting. One of the worst films I have seen.",
    "A complete disaster of a film. The script is an incoherent mess with no clear direction. The pacing is atrocious and the performances are laughably bad. Avoid at all costs.",
    "Dreadful in every conceivable way. The story is derivative and poorly executed. The acting ranges from mediocre to outright terrible. A monumental waste of everyone's time.",
    "I cannot believe this film was actually released. The writing is abysmal and the direction is nonexistent. The cast seem to have no idea what they are doing. Truly awful.",
    "An absolute catastrophe of filmmaking. The plot holes are enormous. The dialogue is cringe-worthy at every turn. The ending makes no sense whatsoever. Profoundly disappointing.",
    "Excruciatingly dull and completely devoid of originality. The film drags interminably through scene after identical scene. The performances are uniformly terrible. A total waste.",
    "One of the most boring films ever committed to screen. Nothing happens for two hours and then the film just ends. The cast seem embarrassed to be there. Avoid entirely.",
    "Lazy, cynical filmmaking that insults the audience's intelligence. The plot is a collection of tired clichés. The direction is flat and uninspired. A deeply poor film.",
    "A crushing disappointment that fails on every level. The script is a first-draft mess that should never have been greenlit. The performances are painfully flat. Dreadful.",
    "Clunky, clichéd, and deeply unpleasant to watch. The story is implausible from the first scene. The characters are stereotypes. The direction is anonymous. A bad film.",
    "Genuinely offensive in its stupidity. The plot asks you to swallow increasingly absurd developments with no internal logic. The cast clearly knew they were in trouble. Dire.",
    "A film so bad it becomes a kind of achievement. The script is awful. The direction is worse. The performances are embarrassing. A complete and utter disaster.",
    "Plotless, pointless, and profoundly irritating. The film has no idea what it wants to be. The characters behave in ways no human ever would. Thoroughly bad cinema.",
    "An embarrassment to everyone involved. The writing is lazy and derivative. The direction is inept. The performances are wooden. A film best forgotten immediately.",
    "Dull, pretentious, and completely devoid of entertainment value. The director mistakes obscurity for depth. The performances are as flat as the script. Thoroughly mediocre.",
    "A terrible film that wastes a potentially interesting premise. The execution is shockingly poor. The pacing is ruinous. The dialogue is clunky and unnatural. A big letdown.",
    "Simply dreadful from start to finish. The film is poorly plotted, badly acted, and incompetently directed. There is not a single scene that works. Avoid without question.",
    "An unmitigated disaster. The story is incoherent. The characters are unlikeable for no good reason. The direction is anonymous and lifeless. A truly terrible film.",
    "Boring beyond belief. The film shuffles through familiar beats with no energy or conviction. The performances are flat. The direction is nonexistent. A complete failure.",
    "A hollow, noisy, and utterly tedious experience. The script is a collection of action movie clichés with zero originality. The performances are robotic. Deeply poor.",
    "One of the most poorly constructed films in recent memory. The narrative is a shambles. The characters are ciphers. The ending is infuriating. A waste of everyone's time.",
    "Excruciating. Every scene is worse than the last. The script has no idea how people actually speak or behave. The cast look mortified. A deeply unpleasant experience.",
    "A film that mistakes noise for excitement and length for substance. The action sequences are incomprehensible. The characters are cardboard. The story is nonexistent. Poor.",
    "Practically unwatchable. The first act goes on for an eternity. The second act is chaotic and confusing. The third act makes no sense. A total creative failure.",
    "Joyless, soulless, and spectacularly boring. The director has nothing to say and takes far too long saying it. The cast clearly gave up early on. A terrible film.",
    "A film that squanders every opportunity it creates. Good moments are immediately undermined by bad choices. The performances are uneven at best. A frustrating failure.",
    "Hideously plotted and terribly acted. The film makes you actively dislike its characters within minutes and never recovers. The direction is flat and lifeless. Poor cinema.",
    "A catastrophic misfire on every level. The comedic timing is off. The dramatic moments fall flat. The action is chaotic and poorly edited. An all-round failure.",
    "Deeply unpleasant and difficult to sit through. The story is unpleasant without purpose or payoff. The performances are unappealing. The direction is mean-spirited. Poor.",
    "A film that tests the patience to its absolute limit. The story is absurdly overcomplicated for no reason. The cast is wasted. The climax is bewildering. A failure.",
    "Cheap, lazy, and profoundly uninspired. The film is content to coast on formula without adding a single original idea. The performances are perfunctory. Skip it entirely.",
    "An agonisingly boring film with nothing to recommend it. The story is both predictable and confusing, an impressive double failure. The cast look lost. Terrible.",
    "A film that feels like it was made by committee and checked against every genre checklist. No spark, no life, no purpose. The cast deserve better. A thorough waste.",
    "Dismal storytelling allied to flat direction and lifeless performances. The film never comes close to achieving what it is clearly aiming for. A complete disappointment.",
    "Muddled, incoherent, and tedious beyond endurance. The script has not the faintest idea what it is about. The performances are uniform in their flatness. Dire.",
    "A film so formulaic it barely registers as cinema. Every beat is telegraphed. Every character is a cliché. Every scene is predictable. The cast phone it in. Poor.",
    "Utterly charmless and self-satisfied. The film confuses cynicism with sophistication. The characters are obnoxious. The story is pointlessly grim. A very poor film indeed.",
    "A complete creative failure dressed up with technical competence. The film looks fine but has absolutely nothing to say. The cast are wasted entirely. Disappointing.",
    "Laborious and joyless. The film grinds forward with grim determination towards a payoff that never arrives. The direction is leaden. The script is a chore. Very poor.",
    "An overlong, overblown, and deeply underwhelming film. The story is stretched far beyond its natural length. The performances are erratic. The ending is unsatisfying. Poor.",
    "A film that takes itself far too seriously and delivers far too little. The portentous direction matches the ponderous script. The cast struggle throughout. A letdown.",
    "Shocking in its badness. The dialogue is some of the worst ever committed to film. The story makes no logical sense. The performances are painful to watch. Avoid.",
    "A meagre and dispiriting effort. The film has no confidence in its own story and hedges every bet. The result is colourless and forgettable. Thoroughly mediocre.",
    "An incoherent mess that improves not at all across its running time. The characters are impossible to care about. The direction is chaotic. The script is a disaster.",
    "Profoundly dull and entirely predictable. The film covers ground that has been covered a thousand times before and adds nothing. The cast are wasted. Poor cinema.",
    "A film without a single original thought in its head. The script is plagiarised from better films. The performances are going through the motions. A complete waste.",
    "Tedious, mean-spirited, and artistically bankrupt. The story is unpleasant and the film is not interested in making sense of it. The direction is cold. Very poor.",
    "A film that mistakes style for substance at every turn. The visuals are slick but empty. The story is hollow. The characters are thin. The whole is deeply unsatisfying.",
    "One of those films where everything that could go wrong does go wrong. The script, the direction, the performances, the editing — all substandard. A collective failure.",
    "Dreary, predictable, and completely forgettable. The film moves through its story with no energy and no conviction. The cast seem disengaged. A waste of their talents.",
    "An uninvolving and deeply tedious experience. The story holds no surprises and generates no tension. The performances are flat. The direction is anonymous. Poor.",
    "Lethargic and charmless. The film has no sense of its own purpose. The characters behave randomly. The climax is preposterous. The cast are entirely wasted. Very bad.",
    "A film that squanders a large budget on a poverty of ideas. The spectacle is impressive but meaningless. The story has no heart. The performances are robotic. Poor.",
    "Crushingly disappointing. The premise promised so much and the film delivers almost nothing. The script is a mess. The direction is uninspired. The cast are let down. Poor.",
    "A tone-deaf film that mistakes offensiveness for edginess. The story goes nowhere interesting. The characters are unpleasant without purpose. The direction is flat. Avoid.",
    "Limp, listless, and deeply unimpressive. The film has no idea how to build tension or sustain interest. The performances range from poor to actively bad. Very poor.",
    "A film that mistakes runtime for importance. Nothing of substance happens for the first ninety minutes. The performances are barely present. The script is a chore. Bad.",
    "Dreary beyond description. The film plods from scene to scene with no momentum or direction. The cast look as bored as the audience will be. A terrible film.",
    "An unfocused and deeply unsatisfying film. The story tries to do too many things and succeeds at none of them. The cast is squandered. The direction is at sea. Poor.",
    "A film bereft of ideas, energy, and conviction. The script is a first draft that should never have left the office. The performances are going through the motions. Bad.",
    "Barely coherent and extremely dull. The film has no discernible plot for its first hour. The second hour is marginally worse. The cast is wasted. A complete failure.",
    "A film in which nothing makes sense and nobody seems to care. The story is arbitrary. The characters are blank. The direction is lifeless. The whole thing is bad.",
    "Infuriating in its incompetence. The story abandons its own rules whenever convenient. The characters behave absurdly. The ending is an insult. A very bad film.",
    "A film that seems designed to alienate its audience at every turn. The tone is off. The pacing is broken. The performances are uneven. The climax is disastrous. Poor.",
    "Charmless and mirthless. The film tries to be funny and fails completely. It tries to be moving and falls flat. The cast cannot salvage the material. A bad film.",
    "A film without grace, wit, or intelligence. The writing is clunky. The direction is flat. The performances are stilted. The story goes nowhere. A thorough failure.",
    "So deeply mediocre it barely registers. The film occupies your attention while it is running and disappears without trace the moment it ends. A waste of time and talent.",
    "A film that generates active dislike rather than mere indifference. The characters are unappealing. The story is unpleasant. The direction is cold. Very poor indeed.",
    "One of the most tiresome films in recent years. The pacing is ruinous. The story is slight and uninteresting. The performances are barely adequate. Thoroughly bad.",
    "A film that mistakes grimness for seriousness. It is relentlessly unpleasant without ever achieving catharsis. The performances suffer. The direction is bleak. Poor.",
    "Interminable and pointless. The film outstays its welcome by approximately ninety minutes. The story could be told in twenty. The cast are duly wasted. Very bad.",
    "An almost heroically bad film. The script is dreadful. The direction is worse. The performances are painful. The whole experience is an extended endurance test. Avoid.",
    "A film that has forgotten what films are for. It offers no pleasure, no insight, no excitement, no emotion. The cast look mortified. A complete artistic failure.",
    "Dreary and derivative in equal measure. The film adds nothing to a genre already groaning with better examples. The cast are going through the motions. Very poor.",
    "A film that collapses under the weight of its own pretensions. The director mistakes obscurantism for depth. The story is impenetrable. The cast look confused. Poor.",
    "Absolutely terrible. The worst film I have seen this year by a considerable margin. The script is an embarrassment. The direction is nonexistent. The cast deserve better.",
    "A film so bad it actively made me angry. The wasted potential is infuriating. The execrable script. The lifeless direction. The flailing performances. An utter failure.",
    "Complete rubbish from start to finish. No engaging characters, no compelling story, no discernible visual style. How this was greenlit remains a mystery. Dreadful.",
    "A film that makes you feel worse for having watched it. Unpleasant, boring, and badly made in equal measure. The cast is wasted. The story is pointless. Poor.",
    "Absolutely nothing works in this film. The story is broken, the direction inept, the performances unconvincing. A failure of filmmaking on every conceivable level.",
    "One of the great wasted opportunities of recent cinema. Everything was in place for a good film and the result is a disaster. The script is the chief culprit. Very poor.",
    "A film that mistakes ugliness for honesty and boredom for realism. The result is unwatchable. The performances cannot save a broken script. A complete failure.",
    "Excruciating from first to last. The film has no redeeming qualities whatsoever. The writing is bad. The direction is bad. The performances are bad. Everything is bad.",
    "A wholly dispiriting experience. The film never finds a reason for its own existence. The cast look lost. The story goes nowhere. The ending is a non-event. Very poor.",
    "Clumsy, charmless, and deeply unconvincing. The film falls at every hurdle. The script is weak, the direction worse, the performances painful. An absolute disaster.",
    "One of the most boring films I have ever been forced to endure. Two hours of my life I will never recover. Avoid at absolutely any cost. A monumental waste of time.",
    "A film of numbing incompetence. Every department has failed. The script, the direction, the performances, the editing — all well below any acceptable standard. Poor.",
    "A depressing reminder of how badly things can go wrong in filmmaking. The script is fundamentally broken. Everything else follows suit. A complete and utter failure.",
    "Awful at every level. I genuinely struggled to stay awake. The film has no momentum, no purpose, and no merit. The cast is blameless but powerless. Avoid.",
    "A film that fills its running time without ever justifying it. Nothing of consequence happens. The characters exist without purpose. The direction is absent. Very bad.",
    "Dismal. A film so comprehensively terrible that it becomes notable in its own right. The script is a catastrophe. The direction is hopeless. The cast are stranded. Poor.",
    "An unmitigated failure of imagination and execution. The film has nothing to offer any audience. The performances are flat. The story is nonexistent. Avoid entirely.",
    "Genuinely dreadful cinema. The kind of film that makes you question why you bother. Everything is wrong. The script, the direction, the performances. All terrible.",
    "A film of spectacular mediocrity. It achieves nothing, says nothing, and leaves you feeling nothing. The cast deserve far better than this. A thorough disappointment.",
    "Broken, boring, and bleak in all the wrong ways. The film is a failure of storytelling, direction, and performance simultaneously. A comprehensive disaster. Very poor.",
    "One of the most excruciating films in living memory. The experience of watching it is actively unpleasant. The cast look mortified. The script is an embarrassment.",
    "A film so thoroughly devoid of quality that it functions as a kind of negative achievement. Awful in every dimension. The cast is wasted. The story is nonexistent. Avoid.",
    "Absolute garbage. I cannot think of a single positive thing to say about this film. It fails at the most basic level of storytelling and continues to fail from there.",
    "A complete and total failure. The worst of everything: writing, direction, acting, editing. The film manages to be both offensively bad and excruciatingly dull. Avoid.",
    "Simply one of the worst films I have ever seen. A collection of bad decisions at every level of production. The cast cannot be blamed for the script. A disaster.",
    "Nothing works. Not the story. Not the characters. Not the direction. Not the performances. The film is a failure on every conceivable level. Very strongly avoid.",
    "A film of utterly catastrophic quality. It insults the audience, wastes its cast, and squanders its budget on a script that should never have been filmed. Dreadful.",
    "An absolute masterclass in how not to make a film. Every mistake that can be made is made here, repeatedly and without any apparent awareness. A complete disaster.",
    "Terrible, terrible, terrible. That is all that needs to be said. The worst film of the year by a wide margin. A total creative catastrophe from start to finish.",
    "Beyond bad. This film transcends ordinary badness to achieve something almost sublime in its failure. Avoid unless you enjoy cinematic punishment. Absolutely dreadful.",
    "One of the most professionally embarrassing films ever released. How anyone approved this script is impossible to understand. The result speaks for itself. Very poor.",
    "A film that has no business existing. It adds nothing, says nothing, achieves nothing. The cast is wasted. The story is broken. The direction is absent. Avoid entirely.",
    "The worst film I have seen in many years. A comprehensive failure at every level. No redeeming qualities whatsoever. A waste of time, money, and talent. Dreadful.",
    "An utter abomination of cinema. Every single element of this production is substandard. Avoid at all costs and warn your friends. A complete and total disaster.",
    "Completely and irredeemably terrible. The film manages to be boring, unpleasant, and badly made simultaneously. The cast look humiliated. A thorough failure. Avoid.",
    "A film of extraordinary badness. The kind of film that makes film school students feel better about their work. Everything here is wrong. Everything. Absolutely dreadful.",
    "So bad as to be genuinely memorable for all the wrong reasons. The script is a catastrophe. The direction is a disaster. The performances are uniformly terrible. Awful.",
    "An extended exercise in cinematic disappointment. Nothing is redeemed. Nothing improves. The film ends and you feel only relief that it is over. Very poor indeed. Avoid.",
    "Completely, comprehensively, spectacularly bad. A film that fails so thoroughly it achieves a kind of perverse greatness in its awfulness. Avoid entirely. Dreadful.",
    "One of the most catastrophically bad films of recent memory. The writing is hopeless. The direction is nonexistent. The performances are excruciating. A total failure.",
    "A disaster so complete that it becomes almost admirable. Every element of filmmaking has been executed as poorly as possible. The cast is blameless. The rest is awful.",
]


def generate_synthetic_reviews(n_positive=250, n_negative=250, seed=42):
    """Generate synthetic movie reviews if CSV file not found."""
    rng = random.Random(seed)
    reviews = []
    pos_pool = POSITIVE_REVIEWS
    neg_pool = NEGATIVE_REVIEWS
    # Positive
    for i in range(n_positive):
        reviews.append({
            "review_id": f"syn_{i+1:04d}",
            "movie_id": f"M{rng.randint(1, 50):03d}",
            "review_text": rng.choice(pos_pool),
            "label": "positive",
        })
    # Negative
    for i in range(n_negative):
        reviews.append({
            "review_id": f"syn_{n_positive+i+1:04d}",
            "movie_id": f"M{rng.randint(1, 50):03d}",
            "review_text": rng.choice(neg_pool),
            "label": "negative",
        })
    rng.shuffle(reviews)
    return pd.DataFrame(reviews)


def load_reviews():
    """Load reviews from CSV or generate synthetic ones."""
    if REVIEWS_CSV.exists():
        logger.info(f"Loading reviews from {REVIEWS_CSV}")
        df = pd.read_csv(REVIEWS_CSV)
        # Ensure required columns
        if "review_text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'review_text' and 'label' columns")
        df = df[["review_text", "label"]].dropna()
        logger.info(f"Loaded {len(df)} reviews from CSV.")
    else:
        logger.info("sample_reviews.csv not found — generating 500 synthetic reviews.")
        df = generate_synthetic_reviews()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(REVIEWS_CSV, index=False)
        logger.info(f"Synthetic reviews saved to {REVIEWS_CSV}")
    return df


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------
def main():
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
    except ImportError as exc:
        logger.error(
            f"Required package not installed: {exc}. "
            "Install with: pip install torch transformers scikit-learn"
        )
        # Still create empty output files so downstream code doesn't break
        METRICS_JSON.write_text(json.dumps({"error": str(exc), "status": "skipped"}))
        SCORES_CSV.write_text("review_text,label,predicted_label,score\n")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    df = load_reviews()
    label2id = {"positive": 1, "negative": 0}
    df["label_id"] = df["label"].map(label2id)

    texts = df["review_text"].tolist()
    labels = df["label_id"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    logger.info(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    class ReviewDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch_results = []
    best_val_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels_batch.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average="binary")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="binary")
        val_precision = precision_score(val_true, val_preds, average="binary")
        val_recall = recall_score(val_true, val_preds, average="binary")

        result = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "train_f1": round(train_f1, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_precision": round(val_precision, 4),
            "val_recall": round(val_recall, 4),
            "val_f1": round(val_f1, 4),
        }
        epoch_results.append(result)
        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(TOKENIZER_DIR))
            logger.info(f"  -> Best model saved (val_f1={val_f1:.4f})")

    # ------------------------------------------------------------------
    # Final evaluation metrics
    # ------------------------------------------------------------------
    cm = confusion_matrix(val_true, val_preds).tolist()
    final_acc = accuracy_score(val_true, val_preds)
    final_precision = precision_score(val_true, val_preds, average="binary")
    final_recall = recall_score(val_true, val_preds, average="binary")
    final_f1 = f1_score(val_true, val_preds, average="binary")

    metrics = {
        "accuracy": round(final_acc, 4),
        "precision": round(final_precision, 4),
        "recall": round(final_recall, 4),
        "f1": round(final_f1, 4),
        "confusion_matrix": cm,
        "label_mapping": {"0": "negative", "1": "positive"},
        "epoch_results": epoch_results,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {METRICS_JSON}")
    logger.info(
        f"Final Val — Accuracy: {final_acc:.4f}, Precision: {final_precision:.4f}, "
        f"Recall: {final_recall:.4f}, F1: {final_f1:.4f}"
    )

    # ------------------------------------------------------------------
    # Score all reviews and save to review_scores.csv
    # ------------------------------------------------------------------
    logger.info("Scoring all reviews with the trained model...")
    model.eval()
    all_texts = df["review_text"].tolist()
    all_true_labels = df["label"].tolist()

    # Process in batches
    scored_rows = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        id2label = {0: "negative", 1: "positive"}
        for j, (text, true_lbl) in enumerate(
            zip(batch_texts, all_true_labels[i : i + BATCH_SIZE])
        ):
            scored_rows.append(
                {
                    "review_text": text,
                    "true_label": true_lbl,
                    "predicted_label": id2label[preds[j]],
                    "score_negative": round(float(probs[j][0]), 4),
                    "score_positive": round(float(probs[j][1]), 4),
                    "confidence": round(float(probs[j].max()), 4),
                }
            )

    scores_df = pd.DataFrame(scored_rows)
    scores_df.to_csv(SCORES_CSV, index=False)
    logger.info(f"Review scores saved to {SCORES_CSV} ({len(scores_df)} rows)")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
