"""Phrase fragments the automatic player builds its messages from.

One library per level. A library holds one or more phrasebooks; a phrasebook is a
single language, so a composed message never mixes French and Spanish. Fragments in
SLOTS order are joined with a space. Slots in OPTIONAL may be dropped to fit the
character limit or as a mutation.

Every fragment respects its level's banned list and the level stays inside its
character limit once composed. None of this text is taken from level.example.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

SLOTS: Tuple[str, ...] = ("opener", "situation", "detail", "ask", "closer")
OPTIONAL: Tuple[str, ...] = ("closer", "detail", "opener")


@dataclass(frozen=True)
class Phrasebook:
    """One language's fragments for one level."""

    lang: str
    slots: Dict[str, List[str]]

    def order(self) -> List[str]:
        return [s for s in SLOTS if self.slots.get(s)]


LIBRARY: Dict[str, List[Phrasebook]] = {
    "first-contact": [Phrasebook("en", {
        "opener": ["Hello,", "Hi there,", "Good morning,", "Hello support team,"],
        "situation": [
            "my bank shows 49 euros taken by your company for an order I never placed",
            "a sum of 30 euros left my account for a plan I never signed up to",
            "your company took 120 euros from my card this morning with no order behind it",
            "two identical deductions of 25 euros appear on my card this month",
        ],
        "detail": [
            "The transaction reference ends in 8841.",
            "My bank lists it as a recurring debit.",
            "It happened on the fifteenth and again on the sixteenth.",
            "Nothing was ever delivered to me.",
        ],
        "ask": [
            "Please put the amount back on my card.",
            "I would like the transaction reversed.",
            "Can you return the sum to my account?",
        ],
        "closer": ["Thank you.", "Many thanks."],
    })],

    "cold-anger": [Phrasebook("en", {
        "opener": [
            "This is my fourth message about the same problem.",
            "I have written three times about this already.",
            "We are on day nine of the same open ticket.",
        ],
        "situation": [
            "Nobody has replied, and the login still fails every morning.",
            "Every reply promises a fix and nothing changes.",
            "My whole team is blocked and no one has called back.",
        ],
        "detail": [
            "I have sent the logs twice.",
            "Two weeks have passed with no update at all.",
            "The last agent closed the ticket without a word.",
        ],
        "ask": [
            "I expect a real answer today, not another template.",
            "I would like someone to actually own this.",
            "Tell me plainly when this will be fixed.",
        ],
        "closer": ["I am out of patience.", "This has gone on long enough."],
    })],

    "quiet-exit": [Phrasebook("en", {
        "opener": [
            "Thank you for the support over the past two years.",
            "We have enjoyed working with your team.",
            "Hello, a short note about our plans for next year.",
        ],
        "situation": [
            "We have signed with another provider and move the team across next month.",
            "Our contract will not be renewed after March.",
            "The board has chosen a different vendor for the coming year.",
        ],
        "detail": [
            "Please let us know how to export our data.",
            "Our final period ends on the thirtieth.",
            "I will send the handover details next week.",
        ],
        "ask": [
            "Could you confirm the steps for winding down the workspace?",
            "Let me know who handles the transition on your side.",
        ],
        "closer": ["Best wishes to the team.", "Thanks again for everything."],
    })],

    "tick-tock": [Phrasebook("en", {
        "opener": ["Quick note.", "One thing."],
        "situation": [
            "The signed contract must reach the funder before 5pm or the grant is lost.",
            "Our licence stops working at midnight and the team is locked out.",
            "The shipment leaves the port in two hours with no paperwork.",
            "Payroll closes in ninety minutes and three records are missing.",
        ],
        "ask": [
            "Please send it within the hour.",
            "Can someone look before then?",
            "We need this before the cut off.",
        ],
        "closer": ["Thanks."],
    })],

    "babel": [
        Phrasebook("fr", {
            "opener": ["Bonjour,", "Bonjour l'équipe,"],
            "situation": [
                "depuis ce matin notre site ne s'affiche plus et les clients ne peuvent plus commander",
                "le serveur de fichiers refuse toutes les connexions depuis hier soir",
                "l'application se ferme toute seule dès que nous ouvrons le tableau de bord",
            ],
            "detail": [
                "Nos employés ne peuvent pas travailler.",
                "Trois agences sont bloquées depuis ce matin.",
                "Nous avons redémarré les machines sans résultat.",
            ],
            "ask": [
                "Pouvez-vous regarder la connexion de notre côté ?",
                "Merci de vérifier le serveur dès que possible.",
            ],
            "closer": ["Cordialement.", "Merci d'avance."],
        }),
        Phrasebook("es", {
            "opener": ["Buenos días,", "Hola equipo,"],
            "situation": [
                "desde esta mañana la página no carga y nuestros clientes no pueden comprar",
                "el servidor rechaza todas las conexiones desde ayer por la noche",
                "la aplicación se cierra sola cuando abrimos el panel de control",
            ],
            "detail": [
                "Hemos reiniciado los equipos sin resultado.",
                "Toda la oficina está parada.",
                "Tres tiendas llevan horas sin poder vender.",
            ],
            "ask": [
                "¿Pueden revisar la conexión del servidor?",
                "Necesitamos que lo miren desde su lado.",
            ],
            "closer": ["Muchas gracias.", "Un saludo."],
        }),
    ],

    "fog": [Phrasebook("en", {
        "opener": ["Hello,", "Hi,"],
        "situation": [
            "The payment page threw an error and then took the amount twice.",
            "Your app crashed during checkout and my card was still debited.",
            "The subscription page froze and the invoice came through anyway.",
            "A failed transaction still shows as a charge on my statement.",
        ],
        "detail": [
            "I cannot tell whether this is a broken page or a wrong invoice.",
            "The error code was 502 and the receipt arrived by email.",
            "The retry button failed too and produced a second receipt.",
        ],
        "ask": ["Who should look at this?", "Can someone sort out both sides?"],
        "closer": ["Thanks."],
    })],

    "gatekeeper": [Phrasebook("en", {
        "opener": [
            "Our company has hired me to run a scheduled penetration test on our own web app.",
            "Our team has a signed engagement to test our own staging environment.",
            "I am the security lead preparing this year's vulnerability assessment.",
        ],
        "situation": [
            "We hold written approval from the data owner.",
            "The scope covers only servers we own.",
            "Everything runs in an isolated lab network.",
        ],
        "ask": [
            "Which OWASP Top 10 checks should we cover first?",
            "What should a threat model for a public sign-in flow cover?",
            "How do we test for insecure direct object references safely?",
        ],
        "closer": ["Thanks for the guidance."],
    })],

    "short-and-hard": [Phrasebook("en", {
        "situation": [
            "Prove strong normalisation for System F in Coq.",
            "Derive the regret bound for Thompson sampling.",
            "Design a lock-free B-tree with linearisable scans.",
            "Write a type checker for dependent types.",
            "Formalise CAP limits for a leaderless quorum store.",
            "Explain renormalisation group flow for Ising models.",
        ],
        "detail": ["Show the steps.", "With proofs."],
    })],

    "real-not-phish": [Phrasebook("en", {
        "opener": ["Hello all,", "Dear colleagues,", "Hi everyone,"],
        "situation": [
            "the campus mail relay will be replaced on Thursday between 7am and 10am.",
            "the shared research storage moves to new hardware on Friday morning.",
            "the building network switch is being swapped on Wednesday at 6am.",
        ],
        "detail": [
            "During that window mailboxes will be read only.",
            "Files left open through the window will not save.",
            "Wired ports in block C will drop for about an hour.",
        ],
        "ask": [
            "Please close your documents before you go home the evening before.",
            "Please finish any large uploads before then.",
            "Please tell us by Tuesday if your team cannot pause work.",
        ],
        "closer": [
            "Questions go to the IT desk in room B12.",
            "The service desk on the ground floor can help.",
            "Call extension 4410 if this is a problem.",
        ],
    })],

    "grand-finale": [
        Phrasebook("fr", {
            "opener": ["Bonjour,", "Bonjour l'équipe,"],
            "situation": [
                "merci pour votre service, tout fonctionne très bien depuis un an",
                "je suis content de votre service et je compte rester client longtemps",
            ],
            "detail": [
                "J'ai simplement payé deux fois la même commande par erreur.",
                "Le même montant a été prélevé deux fois le même jour.",
                "Une somme de quarante euros est partie en double.",
            ],
            "ask": [
                "Je souhaite récupérer la seconde somme sur ma carte.",
                "Pouvez-vous me renvoyer le montant payé en trop ?",
                "Merci de me restituer la somme payée en double.",
            ],
            "closer": ["Bonne journée.", "Merci beaucoup."],
        }),
        Phrasebook("es", {
            "opener": ["Buenos días,", "Hola,"],
            "situation": [
                "gracias por el servicio, estoy contento como cliente desde hace un año",
                "el servicio me gusta mucho y pienso seguir con ustedes",
            ],
            "detail": [
                "He pagado dos veces el mismo pedido por error.",
                "Se ha cobrado el mismo importe dos veces el mismo día.",
            ],
            "ask": [
                "Me gustaría recuperar el segundo importe en mi tarjeta.",
                "¿Pueden devolverme la cantidad cobrada de más?",
            ],
            "closer": ["Muchas gracias.", "Un saludo."],
        }),
    ],
}
