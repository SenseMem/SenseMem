"""Reading order and original-pixel coordinates; these specifications never redraw a figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Point = tuple[int, int]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Signal:
    points: tuple[Point, ...]
    start: float = 0.2
    end: float = 1.5
    radius: int = 4
    flow: bool = True


@dataclass(frozen=True)
class Cue:
    boxes: tuple[Box, ...]
    start: float = 0.2
    end: float = 2.5
    direction: Literal["still", "right", "down"] = "still"


@dataclass(frozen=True)
class Step:
    title: str
    explanation: str
    focus: tuple[Box, ...] = ()
    signals: tuple[Signal, ...] = ()
    seconds: float = 3.6
    cues: tuple[Cue, ...] = ()
    overview: bool = False


LIFECYCLE_OVERVIEW = Step(
    "Native experience -> revisable knowledge -> task context",
    "At 09:10 the desk observation completes its lifecycle; the later 10:20 observation revises it to the cabinet before retrieval.",
    seconds=1.6,
    overview=True,
)


def _lifecycle_capture(observation: Literal[1, 2]) -> tuple[Step, ...]:
    """Repeat the original capture arrows once for each chronological observation."""
    if observation == 1:
        timestamp = "09:10"
        input_box = (22, 225, 176, 303)
        observation_box = (223, 148, 419, 313)
        retained_box = (628, 577, 857, 718)
        input_path = ((188, 325), (200, 325), (200, 238), (212, 238))
        commit_path = ((429, 238), (445, 238), (445, 325), (463, 325))
    else:
        timestamp = "10:20"
        input_box = (22, 314, 176, 385)
        observation_box = (223, 331, 419, 474)
        retained_box = (875, 578, 1119, 719)
        input_path = ((188, 325), (200, 325), (200, 400), (212, 400))
        commit_path = ((429, 400), (445, 400), (445, 325), (463, 325))
    return (
        Step(
            f"{timestamp}: accept observation o_{observation}",
            "The native input becomes this observation. The later observation does not arrive during the first capture.",
            focus=(input_box,),
            signals=(Signal(input_path, 0.55, 1.05),),
            seconds=1.8,
            cues=(
                Cue((input_box,), 0.1, 0.5),
                Cue((observation_box,), 1.1, 1.7, "down"),
            ),
        ),
        Step(
            f"{timestamp}: commit o_{observation}, then show its retained source and ACK together",
            "The commit durably retains the native source before inference. The retained-source view and ACK respond in parallel to that same completed commit.",
            focus=(observation_box,),
            signals=(
                Signal(commit_path, 0.65, 1.25),
                Signal(((581, 319), (591, 319), (591, 570), (607, 570)), 1.9, 2.5),
                Signal(((522, 390), (522, 453)), 1.9, 2.5),
            ),
            seconds=4.0,
            cues=(
                Cue((observation_box,), 0.1, 0.6),
                Cue(((464, 260, 581, 353),), 1.3, 1.85, "down"),
                Cue((retained_box, (472, 456, 571, 498)), 2.55, 3.9, "down"),
            ),
        ),
        Step(
            f"{timestamp}: wait for host scheduling",
            "Hold ACK: capture does not itself schedule settlement.",
            seconds=0.5,
        ),
        Step(
            f"{timestamp}: host settlement, encoding, and index projection",
            "After source commit the host calls settle; encoding precedes the Zvec update. SQLite index work commits before the derived projection.",
            focus=((464, 260, 581, 353),),
            signals=(
                Signal(((451, 353), (451, 553), (72, 553), (72, 579)), 0.6, 1.9, radius=3),
                Signal(((120, 649), (138, 649)), 2.55, 2.95),
                Signal(((237, 649), (254, 649)), 4.2, 4.6),
                Signal(((523, 828), (641, 828)), 4.2, 4.9),
            ),
            seconds=6.4,
            cues=(
                Cue(((464, 260, 581, 353),), 0.1, 0.5),
                Cue(((24, 584, 119, 714),), 1.95, 2.5, "down"),
                Cue(((141, 584, 235, 714),), 3.0, 3.55, "down"),
                Cue(((24, 799, 509, 846),), 3.55, 4.15, "right"),
                Cue(((257, 584, 351, 714), (666, 799, 1009, 846)), 4.95, 6.25, "down"),
                Cue(((1048, 802, 1641, 844),), 5.5, 6.25, "right"),
            ),
        ),
        Step(
            f"{timestamp}: optional proposal and validation",
            "After projection, eligible formation may propose an interpretation. Validation precedes acceptance into the illustrated knowledge families; icons name roles rather than a literal model network.",
            focus=((257, 584, 351, 714),),
            signals=(
                Signal(((351, 649), (368, 649)), 0.6, 0.95),
                Signal(((467, 649), (481, 649)), 1.6, 1.95),
                Signal(((579, 649), (604, 649), (604, 182), (609, 182)), 2.6, 3.5),
            ),
            seconds=4.6,
            cues=(
                Cue(((257, 584, 351, 714),), 0.1, 0.55),
                Cue(((371, 584, 465, 714),), 1.0, 1.55, "down"),
                Cue(((484, 584, 579, 714),), 2.0, 2.55, "down"),
                Cue(((616, 164, 1127, 202),), 3.55, 3.95, "right"),
                Cue(((615, 74, 1125, 140),), 4.0, 4.45, "right"),
            ),
        ),
    )


LIFECYCLE = (
    Step(
        "Accept every native modality",
        "Text, image, audio, and video are parallel native-input capabilities; the four rows do not form a processing sequence.",
        focus=((10, 92, 188, 482),),
        seconds=2.8,
        cues=(Cue(((10, 92, 188, 482),), 0.1, 2.7),),
    ),
    *_lifecycle_capture(1),
    Step(
        "09:10: accepted desk interpretation with its image support",
        "Only the first source has arrived. Its accepted desk interpretation has a support relationship, not another execution pipeline; no future endpoint is highlighted yet.",
        focus=((628, 577, 857, 718),),
        signals=(Signal(((621, 648), (621, 419), (710, 419), (710, 407)), 0.75, 1.45, flow=False),),
        seconds=2.4,
        cues=(
            Cue(((628, 577, 857, 718),), 0.1, 0.7),
            Cue(((628, 218, 839, 404),), 1.5, 2.35, "down"),
        ),
    ),
    *_lifecycle_capture(2),
    Step(
        "10:20: revise the existing desk interpretation with new support",
        "The existing desk version and newly retained o_2 meet at the cabinet revision. The result first highlights after both original incoming connections complete.",
        focus=((628, 218, 839, 404), (875, 578, 1119, 719)),
        signals=(
            Signal(((1123, 648), (1123, 419), (1090, 419), (1090, 407)), 0.9, 1.8, flow=False),
            Signal(((846, 321), (894, 321)), 0.9, 1.8),
        ),
        seconds=3.5,
        cues=(
            Cue(((875, 578, 1119, 719),), 0.1, 0.8),
            Cue(((628, 218, 839, 404),), 0.1, 0.8),
            Cue(((900, 218, 1116, 404),), 1.9, 3.4, "down"),
        ),
    ),
    Step(
        "10:20: close the past interval and retain current world time",
        "Only after the cabinet revision is accepted does the animation emphasize the desk interval ending at 10:20 and the cabinet interval continuing from 10:20.",
        focus=((900, 218, 1116, 404),),
        signals=(Signal(((944, 472), (1106, 472)), 0.7, 1.3),),
        seconds=2.5,
        cues=(
            Cue(((900, 218, 1116, 404),), 0.1, 0.6),
            Cue(((630, 448, 1117, 508),), 1.4, 2.4, "right"),
        ),
    ),
    Step(
        "The current claim and its source form the evidence closure",
        "Both originals remain retained. The highlighted current evidence closure consists of the cabinet interpretation and its 10:20 source.",
        focus=((628, 577, 857, 718), (875, 578, 1119, 719), (900, 218, 1116, 404)),
        seconds=2.2,
        cues=(
            Cue(((628, 577, 857, 718), (875, 578, 1119, 719)), 0.1, 0.65),
            Cue(((900, 218, 1116, 404),), 0.7, 1.2),
            Cue(((870, 208, 1127, 750),), 1.3, 2.1),
        ),
    ),
    Step(
        "A task query searches dense and lexical indexes",
        "The headset question first finds candidate IDs through dense and lexical lookup. An index match alone is not authoritative evidence.",
        focus=((1167, 84, 1646, 154),),
        signals=(Signal(((1410, 154), (1410, 174), (1256, 174), (1256, 195)), 0.7, 1.4),),
        seconds=3.8,
        cues=(
            Cue(((1167, 84, 1646, 154),), 0.1, 0.65, "right"),
            Cue(((1167, 201, 1344, 237),), 1.45, 1.9),
            Cue(((1179, 240, 1253, 316), (1267, 240, 1332, 316)), 2.0, 3.65, "down"),
        ),
    ),
    Step(
        "Hydrate, filter, and complete support",
        "SQLite hydration drops stale index IDs and applies visibility and scope. The eligible cabinet claim must retain its complete supporting observation.",
        focus=((1167, 201, 1344, 328),),
        signals=(
            Signal(((1345, 268), (1364, 268)), 0.6, 1.0),
            Signal(((1481, 268), (1498, 268)), 1.65, 2.05),
        ),
        seconds=3.6,
        cues=(
            Cue(((1167, 201, 1344, 328),), 0.1, 0.55),
            Cue(((1369, 201, 1479, 328),), 1.05, 1.6, "down"),
            Cue(((1502, 201, 1646, 328),), 2.1, 3.45, "down"),
        ),
    ),
    Step(
        "compile() returns supported context within a budget",
        "Compilation selects supported context within B, then returns its records and source certificate together as one output bundle without generating an answer.",
        focus=((1502, 201, 1646, 328),),
        signals=(Signal(((1575, 328), (1575, 357), (1293, 357), (1293, 391)), 0.65, 1.35),),
        seconds=4.0,
        cues=(
            Cue(((1502, 201, 1646, 328),), 0.1, 0.6),
            Cue(((1167, 398, 1429, 444),), 1.4, 1.8),
            Cue(((1244, 447, 1355, 484),), 1.85, 2.25),
            Cue(((1180, 490, 1416, 624), (1180, 634, 1416, 716)), 2.3, 3.9, "down"),
        ),
    ),
    Step("Alternative public call", "An ask example starts independently of compile.", seconds=0.7),
    Step(
        "Alternatively, ask() generates a grounded answer",
        "A separate ask call sends complete support to the configured reader. After generation, the cabinet answer and its claim and source references return together.",
        focus=((1502, 201, 1646, 328),),
        signals=(
            Signal(((1575, 328), (1575, 357), (1548, 357), (1548, 391)), 0.65, 1.35),
            Signal(((1544, 486), (1544, 525)), 2.4, 2.8),
        ),
        seconds=4.5,
        cues=(
            Cue(((1502, 201, 1646, 328),), 0.1, 0.6),
            Cue(((1439, 398, 1646, 444),), 1.4, 1.8),
            Cue(((1462, 451, 1626, 487),), 1.85, 2.35),
            Cue(((1453, 531, 1634, 629), (1463, 635, 1619, 696)), 2.85, 4.4, "down"),
        ),
    ),
    LIFECYCLE_OVERVIEW,
)

SUPPORT_OVERVIEW = Step(
    "Complete support -> sufficient witnesses -> one charge per shared source",
    "Read the original four panels in order. Counts are schematic; character, certificate-text, and media costs also apply.",
    seconds=1.5,
    overview=True,
)
SUPPORT = (
    Step(
        "(a) An AND assessment needs every premise",
        "o_1 and o_2 must both support u. One convenient premise cannot satisfy the joint assessment.",
        ((24, 16, 288, 52), (492, 149, 690, 190)),
        (
            Signal(
                ((120, 375), (120, 364), (126, 351), (141, 345), (173, 345), (173, 343)),
                0.45,
                1.25,
                radius=3,
                flow=False,
            ),
            Signal(
                ((229, 375), (229, 364), (223, 351), (208, 345), (173, 345), (173, 343)),
                0.45,
                1.25,
                radius=3,
                flow=False,
            ),
            Signal(((173, 302), (173, 284)), 1.90, 2.22, radius=3, flow=False),
        ),
        4.8,
        cues=(
            Cue(((80, 374, 160, 425), (188, 374, 269, 425)), 0.12, 4.8),
            Cue(((138, 302, 209, 343),), 1.30, 4.8),
            Cue(((140, 219, 205, 285),), 2.25, 4.8),
            Cue(((492, 149, 690, 190),), 2.45, 4.8, "right"),
            Cue(((454, 238, 711, 271),), 3.05, 4.8, "right"),
        ),
    ),
    Step(
        "(a) An OR chooses an eligible complete branch",
        "The selected closure reaches k through u. The alternative o_3 remains retained, while the chosen AND branch stays whole.",
        (
            (24, 16, 288, 52),
            (307, 228, 386, 278),
            (432, 389, 682, 424),
            (492, 111, 712, 148),
        ),
        (
            Signal(
                ((173, 220), (173, 212), (180, 196), (197, 189), (220, 189)),
                0.60,
                1.25,
                radius=3,
                flow=False,
            ),
            Signal(((252, 168), (252, 148)), 1.80, 2.16, radius=3, flow=False),
        ),
        4.6,
        cues=(
            Cue(
                ((80, 374, 160, 425), (188, 374, 269, 425), (140, 219, 205, 285)),
                0.10,
                4.6,
            ),
            Cue(((221, 168, 284, 210),), 1.30, 4.6),
            Cue(((219, 82, 285, 149),), 2.20, 4.6),
            Cue(((55, 69, 319, 445),), 2.45, 4.6),
            Cue(((492, 111, 712, 148),), 2.70, 4.6, "right"),
            Cue(((431, 316, 705, 384),), 3.20, 4.6, "down"),
        ),
    ),
    Step(
        "Local pause between support and scoring",
        "Hold the completed support example before examining count and confidence.",
        seconds=0.8,
    ),
    Step(
        "(b) Evaluate the two witness groups in parallel",
        "P_1 and P_2 yield n = 2, h = 0.36; the joint P_3 yields n = 1, h = 0.99. The separate calculations retain their own conclusions.",
        (
            (790, 16, 1168, 53),
            (807, 79, 1138, 246),
            (1203, 101, 1515, 147),
            (1220, 165, 1515, 211),
        ),
        seconds=4.8,
        cues=(
            Cue(((807, 79, 1138, 121),), 0.10, 4.8),
            Cue(((819, 121, 1030, 161), (819, 162, 1030, 204)), 0.25, 2.0),
            Cue(((819, 205, 1030, 246),), 0.25, 2.0),
            Cue(((1030, 121, 1137, 161), (1030, 162, 1137, 204)), 1.05, 2.0),
            Cue(((1030, 205, 1137, 246),), 1.05, 2.0),
            Cue(((1203, 101, 1515, 147),), 2.05, 4.8, "right"),
            Cue(((1220, 165, 1515, 211),), 2.05, 4.8, "right"),
        ),
    ),
    Step(
        "(b) Count and confidence can require different witnesses",
        "Combining the displayed witnesses meets n = 2 and h = 0.99; the delivered union still contains only k, o_1, and o_2.",
        ((807, 79, 1138, 246), (1203, 101, 1515, 211)),
        (Signal(((1164, 233), (1164, 278)), 2.05, 2.42, radius=3),),
        5.8,
        cues=(
            Cue(((819, 121, 1137, 204), (1203, 101, 1515, 147)), 0.15, 1.90),
            Cue(((819, 205, 1137, 246), (1220, 165, 1515, 211)), 0.15, 1.90),
            Cue(((882, 285, 1423, 330),), 2.48, 4.05, "right"),
            Cue(((921, 359, 1192, 424), (1238, 359, 1465, 425)), 4.10, 5.8),
        ),
    ),
    Step(
        "Local pause before selecting the delivery set",
        "Hold the witness conclusion before the selected-delivery example.",
        seconds=0.8,
    ),
    Step(
        "(c) Select sufficient complete support",
        "Select the existing claim k with sufficient support from o_1 and o_5. Other audit sources remain retained; no new claim is created.",
        ((24, 477, 406, 511), (364, 517, 430, 581), (80, 648, 708, 699)),
        (
            Signal(((118, 648), (375, 575)), 0.95, 1.90, radius=3, flow=False),
            Signal(((437, 648), (402, 580)), 0.95, 1.90, radius=3, flow=False),
        ),
        4.6,
        cues=(
            Cue(((81, 647, 154, 700), (400, 647, 475, 700)), 0.20, 4.6),
            Cue(((364, 517, 430, 581),), 1.95, 4.6),
            Cue(((112, 700, 443, 745),), 2.15, 4.6, "right"),
            Cue(((158, 749, 475, 783),), 2.70, 4.6, "right"),
        ),
    ),
    Step(
        "(c) Audit references and delivered records are different",
        "The audit union contains nine records. This selected-proof example delivers three; default compilation keeps the full audit union.",
        (
            (80, 648, 708, 699),
            (34, 808, 426, 906),
            (525, 808, 743, 906),
            (364, 517, 430, 581),
        ),
        (Signal(((440, 879), (512, 879)), 2.75, 3.20, radius=3),),
        5.2,
        cues=(
            Cue(((142, 809, 319, 838),), 0.10, 1.50),
            Cue(((48, 857, 413, 904),), 0.45, 1.50, "right"),
            Cue(
                (
                    (48, 857, 96, 904),
                    (103, 858, 141, 903),
                    (260, 858, 297, 903),
                    (81, 647, 154, 700),
                    (400, 647, 475, 700),
                    (364, 517, 430, 581),
                ),
                1.65,
                5.2,
            ),
            Cue(((444, 831, 510, 864),), 1.65, 5.2),
            Cue(((525, 809, 743, 846), (544, 856, 727, 907)), 3.25, 5.2),
        ),
    ),
    Step(
        "Local pause before comparing source costs",
        "Hold the delivered-record group before the item-cost examples.",
        seconds=0.8,
    ),
    Step(
        "(d) Disjoint sources consume separate slots",
        "At B_items = 6, selecting support for two existing claims uses four distinct sources. The dots mark singleton assessments, not claim generation.",
        ((790, 477, 1170, 512), (795, 567, 1123, 842), (1080, 514, 1216, 559)),
        (
            Signal(((844, 730), (888, 663)), 0.65, 1.05, radius=3, flow=False),
            Signal(((917, 730), (888, 663)), 0.65, 1.05, radius=3, flow=False),
            Signal(((1002, 730), (1038, 663)), 0.65, 1.05, radius=3, flow=False),
            Signal(((1075, 730), (1038, 663)), 0.65, 1.05, radius=3, flow=False),
        ),
        5.4,
        cues=(
            Cue(((1080, 514, 1216, 559),), 0.05, 5.4),
            Cue(
                (
                    (813, 728, 886, 773),
                    (886, 728, 960, 773),
                    (968, 728, 1041, 773),
                    (1043, 728, 1117, 773),
                ),
                0.15,
                5.4,
            ),
            Cue(((860, 610, 916, 665), (1010, 610, 1068, 665)), 1.12, 5.4),
            Cue(((850, 777, 1073, 810),), 3.40, 5.4, "right"),
            Cue(((907, 811, 1024, 846),), 3.75, 5.4, "right"),
        ),
    ),
    Step(
        "(d) Several claims can reuse the same full sources",
        "Select support for four existing claims from the same o_1 and o_2 singleton assessments. Sharing those records adds no copies or item charges.",
        ((1153, 566, 1510, 811), (1080, 514, 1216, 559)),
        (
            *(
                Signal(
                    ((source, 750), (claim, 663)),
                    0.65,
                    1.40,
                    radius=3,
                    flow=False,
                )
                for claim in (1218, 1296, 1374, 1452)
                for source in (1261, 1390)
            ),
            Signal(((1332, 847), (1332, 861)), 3.00, 3.30, radius=3),
        ),
        5.0,
        cues=(
            Cue(((1080, 514, 1216, 559),), 0.05, 5.0),
            Cue(((1221, 748, 1302, 793), (1350, 748, 1430, 793)), 0.15, 5.0),
            Cue(
                tuple((claim - 29, 609, claim + 29, 665) for claim in (1218, 1296, 1374, 1452)),
                1.45,
                5.0,
            ),
            Cue(((1204, 796, 1459, 818),), 2.00, 5.0, "right"),
            Cue(((1280, 818, 1389, 848),), 2.50, 5.0),
            Cue(((1274, 860, 1392, 908),), 3.35, 5.0),
            Cue(((1202, 916, 1483, 947),), 3.70, 5.0, "right"),
        ),
    ),
    Step(
        "(d) Compare the delivered context under the same item budget",
        "Disjoint sources permit two claims; shared sources permit four. Both examples consume six unique records.",
        ((795, 567, 1510, 947), (1080, 514, 1216, 559)),
        seconds=3.8,
        cues=(
            Cue(((1080, 514, 1216, 559),), 0.05, 3.8),
            Cue(((907, 811, 1024, 846),), 0.20, 3.8),
            Cue(((1280, 818, 1389, 848),), 0.80, 3.8),
            Cue(((1274, 860, 1392, 908), (1202, 916, 1483, 947)), 1.30, 3.8),
        ),
    ),
    SUPPORT_OVERVIEW,
)

CONTEXT_OVERVIEW = Step(
    "Read the complete architecture",
    "Finish with the unchanged source image; the host controls goals, scheduling, and actions.",
    seconds=3.0,
    overview=True,
)
CONTEXT = (
    Step(
        "Partial-input example: stop at the transient branch",
        "The modality rows describe supported inputs. This example takes only the no branch and stops before commit.",
        ((8, 122, 230, 584),),
        (
            Signal(((328, 221), (354, 221)), 2.05, 2.37),
            Signal(((423, 221), (453, 221)), 2.9, 3.22),
        ),
        4.8,
        (
            Cue(
                (
                    (20, 177, 217, 244),
                    (20, 259, 217, 337),
                    (20, 343, 217, 416),
                    (20, 434, 217, 503),
                ),
                0.15,
                1.65,
                "right",
            ),
            Cue(((277, 205, 328, 237),), 1.72, 2.3),
            Cue(((355, 198, 423, 245), (423, 193, 453, 216)), 2.37, 3.1),
            Cue(((454, 205, 502, 237),), 3.22, 4.5),
        ),
    ),
    Step(
        "Local pause before a separate finalized-input example",
        "Hold the transient branch before following a fresh finalized observation.",
        seconds=0.9,
    ),
    Step(
        "Finalized input: one durable commit, then ACK",
        "O and pending work commit together. ACK confirms capture, without requiring model work or index readiness.",
        ((277, 205, 328, 237),),
        (
            Signal(((328, 221), (354, 221)), 0.7, 1.02),
            Signal(((388, 246), (388, 295)), 1.4, 1.8),
            Signal(((389, 503), (389, 522)), 3.5, 3.82),
        ),
        5.3,
        (
            Cue(((20, 343, 217, 416),), 0.05, 0.4, "right"),
            Cue(((277, 205, 328, 237),), 0.15, 0.9),
            Cue(((355, 198, 423, 245), (398, 248, 433, 274)), 1.02, 1.6),
            Cue(((350, 299, 429, 325),), 1.8, 2.1),
            Cue(
                ((279, 326, 378, 489), (399, 326, 498, 489), (968, 702, 1118, 729)),
                2.1,
                3.5,
                "down",
            ),
            Cue(((348, 523, 434, 555),), 3.82, 5.0),
        ),
    ),
    Step(
        "Local pause at the host scheduling boundary",
        "Hold the capture acknowledgement; ACK does not start the next operation.",
        seconds=1.2,
    ),
    Step(
        "Host-scheduled work uses configured computation roles",
        "The host starts work. Interpret, Plan, and Integrate are capabilities; the network artwork is not an execution graph.",
        ((279, 326, 378, 489),),
        signals=(Signal(((518, 330), (533, 330)), 1.0, 1.32),),
        seconds=4.5,
        cues=(
            Cue(((1505, 503, 1650, 613),), 0.15, 1.0),
            Cue(((550, 418, 670, 476),), 1.32, 2.2),
            Cue(((589, 509, 764, 542),), 1.9, 2.55),
            Cue(((550, 205, 632, 237), (645, 205, 714, 237), (724, 205, 809, 237)), 2.6, 4.3),
        ),
    ),
    Step(
        "Local pause before a separate specialist example",
        "Hold the computation focus; specialist use is conditional.",
        seconds=0.7,
    ),
    Step(
        "Conditional specialist example",
        "ASR illustrates one configured specialist call, not a mandatory pass through every plugin.",
        ((291, 432, 358, 450), (589, 509, 764, 542)),
        (
            Signal(((603, 563), (603, 637)), 1.1, 1.55),
            Signal(((657, 637), (657, 563)), 2.6, 3.05),
        ),
        4.6,
        (
            Cue(((291, 432, 358, 450),), 0.15, 0.9),
            Cue(((589, 509, 764, 542),), 0.65, 1.1),
            Cue(((281, 686, 391, 786),), 1.55, 2.6, "right"),
            Cue(((686, 418, 807, 476),), 3.05, 4.3),
        ),
    ),
    Step(
        "Local pause before resuming source processing",
        "Hold the specialist result before continuing the source-processing example.",
        seconds=0.7,
    ),
    Step(
        "Commit enriched sources before projecting their index",
        "Source records and embeddings commit in SQLite first. Their Zvec projection precedes optional formation.",
        ((866, 670, 1150, 795),),
        (Signal(((1151, 740), (1233, 740)), 1.6, 2.2),),
        4.0,
        (
            Cue(((866, 670, 1150, 699), (896, 712, 947, 774), (968, 702, 1118, 729)), 0.2, 1.7),
            Cue(((1235, 670, 1405, 795),), 2.2, 3.7, "down"),
        ),
    ),
    Step(
        "Optional supported formation: an AND branch",
        "Both source premises are needed for the illustrated AND branch. Type icons describe available knowledge kinds.",
        ((550, 205, 632, 237), (891, 272, 928, 306), (891, 337, 928, 372)),
        (
            Signal(((926, 295), (961, 307)), 1.0, 1.45, flow=False),
            Signal(((924, 345), (963, 322)), 1.0, 1.45, flow=False),
            Signal(((997, 313), (1032, 313)), 2.0, 2.4, flow=False),
        ),
        4.8,
        (
            Cue(((550, 205, 632, 237),), 0.15, 0.8),
            Cue(((891, 272, 928, 306), (891, 337, 928, 372)), 0.55, 1.5),
            Cue(((960, 296, 996, 331),), 1.57, 2.3),
            Cue(((1033, 296, 1069, 331),), 2.4, 3.5),
            Cue(((818, 317, 834, 334), (980, 190, 1040, 246)), 3.2, 4.5),
        ),
    ),
    Step(
        "An alternative complete branch",
        "The OR also permits the O3 branch. The previous AND selection is released before this alternative is shown.",
        ((1011, 350, 1049, 387),),
        (Signal(((1036, 351), (1045, 331)), 0.85, 1.25, flow=False),),
        2.8,
        (
            Cue(((1011, 350, 1049, 387),), 0.35, 1.1),
            Cue(((1033, 296, 1069, 331),), 1.25, 2.5),
        ),
    ),
    Step(
        "One committed state feeds supported memory and the search projection",
        "Records and versions commit together in SQLite. Its supported-memory view and derived search projection then fan out without depending on each other's display.",
        ((866, 670, 1150, 795), (856, 411, 949, 496)),
        (
            Signal(((924, 637), (924, 623)), 1.3, 1.55),
            Signal(((924, 573), (924, 566)), 1.8, 2.1),
            Signal(((1151, 740), (1233, 740)), 1.3, 2.1),
            Signal(((950, 455), (976, 455)), 2.1, 2.55, flow=False),
        ),
        4.2,
        (
            Cue(((968, 734, 1118, 759), (968, 765, 1118, 790)), 0.2, 1.2),
            Cue(((882, 575, 966, 614),), 1.45, 2.15),
            Cue(
                (
                    (856, 411, 949, 496),
                    (981, 411, 1074, 496),
                    (858, 511, 1105, 542),
                    (1235, 670, 1405, 795),
                ),
                2.1,
                3.9,
            ),
        ),
    ),
    Step(
        "Local pause before an independent query example",
        "Hold the supported-memory view; querying does not automatically invoke formation.",
        seconds=1.2,
    ),
    Step(
        "A query and a budget start a separate compile call",
        "The goal and budget configure retrieval. The derived index supplies candidate IDs rather than authoritative delivery.",
        ((1199, 190, 1254, 222), (1300, 190, 1418, 222)),
        (Signal(((1204, 563), (1204, 588), (1036, 588), (1036, 636)), 1.8, 3.0),),
        4.5,
        (
            Cue(((1199, 190, 1254, 222), (1300, 190, 1418, 222)), 0.2, 1.2),
            Cue(((1152, 264, 1233, 322),), 1.2, 1.8),
            Cue(((1235, 670, 1405, 795),), 3.0, 4.2),
        ),
    ),
    Step(
        "Candidates must be hydrated and filtered",
        "SQLite reads determine existence, visibility, and scope. Missing or stale index IDs cannot become delivered evidence.",
        ((1152, 264, 1233, 322), (1235, 670, 1405, 795)),
        (
            Signal(((1194, 323), (1194, 351)), 0.55, 0.9),
            Signal(((1255, 637), (1255, 563)), 2.3, 2.85),
            Signal(((1230, 223), (1230, 238), (1295, 238), (1295, 250)), 2.3, 2.85),
            Signal(((1359, 223), (1359, 238), (1299, 238)), 2.3, 2.85),
            Signal(((1234, 293), (1261, 293)), 2.85, 3.35),
        ),
        4.7,
        (
            Cue(((1152, 264, 1233, 322),), 0.15, 0.65),
            Cue(((1149, 355, 1240, 382),), 0.9, 1.4),
            Cue(((1162, 383, 1223, 414), (1162, 415, 1223, 447)), 1.0, 2.0, "right"),
            Cue(((866, 670, 1150, 795),), 1.8, 2.5),
            Cue(((1199, 190, 1254, 222), (1300, 190, 1418, 222)), 2.0, 3.0),
            Cue(((1263, 265, 1328, 320),), 3.35, 4.4),
        ),
    ),
    Step(
        "Select a complete claim-and-support group",
        "Complete support is read from retained evidence, without creating new claims. Required premises and source union travel together.",
        ((1263, 265, 1328, 320), (1145, 353, 1241, 470)),
        (
            Signal(((1329, 293), (1355, 293)), 0.55, 0.9),
            Signal(((926, 295), (961, 307)), 1.0, 1.4, flow=False),
            Signal(((924, 345), (963, 322)), 1.0, 1.4, flow=False),
            Signal(((997, 313), (1032, 313)), 1.85, 2.2, flow=False),
            Signal(((1118, 325), (1132, 325)), 2.6, 2.92),
            Signal(((1242, 423), (1252, 423)), 3.35, 3.7),
        ),
        5.2,
        (
            Cue(((1263, 265, 1328, 320),), 0.15, 0.7),
            Cue(((1357, 259, 1449, 303),), 0.9, 1.6),
            Cue(((891, 272, 928, 306), (891, 337, 928, 372)), 0.65, 1.5),
            Cue(((960, 296, 996, 331),), 1.52, 1.95),
            Cue(((1033, 296, 1069, 331),), 2.2, 2.7),
            Cue(((1357, 259, 1449, 342),), 2.95, 3.4),
            Cue(((1254, 353, 1373, 470), (1387, 352, 1453, 481)), 3.7, 4.9, "down"),
        ),
    ),
    Step(
        "Pack the supported records within the budget",
        "The bundle includes selected records and their required unique sources. The budget constrains the complete delivered group.",
        ((1254, 353, 1373, 470), (1387, 352, 1453, 481)),
        (
            Signal(((1417, 479), (1417, 489), (1305, 489)), 0.8, 1.4),
            Signal(((1305, 490), (1305, 507)), 1.55, 1.87),
            Signal(((1464, 339), (1494, 339)), 3.1, 3.45),
        ),
        4.3,
        (
            Cue(((1254, 353, 1373, 470), (1387, 352, 1453, 481)), 0.15, 0.9),
            Cue(((1183, 510, 1423, 549), (1300, 190, 1418, 222)), 1.87, 3.3, "right"),
            Cue(((1495, 129, 1658, 173),), 3.45, 4.0),
        ),
    ),
    Step(
        "Return multimodal context to the host",
        "The original output rows illustrate context structure, not a guarantee that every query returns every modality. compile generates no answer.",
        ((1495, 129, 1658, 468),),
        (Signal(((1577, 468), (1577, 501)), 2.25, 2.65),),
        4.2,
        (
            Cue(
                (
                    (1505, 183, 1648, 239),
                    (1505, 247, 1648, 303),
                    (1505, 310, 1648, 366),
                    (1505, 374, 1648, 430),
                ),
                0.15,
                2.2,
                "right",
            ),
            Cue(((1505, 503, 1650, 613),), 2.65, 3.9),
        ),
    ),
    Step(
        "Feedback is available, not automatically executed",
        "A static emphasis acknowledges optional host-controlled maintenance. No motion closes an automatic query-to-revision loop.",
        ((969, 821, 1158, 854), (1505, 503, 1650, 613)),
        seconds=2.3,
        cues=(Cue(((969, 821, 1158, 854),), 0.4, 2.0),),
    ),
    CONTEXT_OVERVIEW,
)

DIAGRAMS = {
    "memory-lifecycle": LIFECYCLE,
    "contextos-overview": CONTEXT,
    "support-compilation": SUPPORT,
}
