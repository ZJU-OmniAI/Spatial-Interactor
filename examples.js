/* Generated from the selected appendix cases. */
Object.assign(window.SPATIAL_DATA, {
  "levels": {
    "l1": {
      "title": "L1: Passive world-state transitions",
      "count": "15,109",
      "description": "Explain external-world changes under an approximately stable viewpoint, from object displacement and articulation to ordered operations and their outcomes.",
      "tasks": [
        "Object displacement",
        "Attribute & articulation changes",
        "Occlusion & visibility",
        "Relative configuration",
        "Single- & multi-step operations"
      ],
      "cases": [
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
        "E09",
        "E10",
        "E28"
      ]
    },
    "l2": {
      "title": "L2: Active self-state transitions",
      "count": "69,487",
      "description": "Attribute observation changes to camera translation, rotation, and elevation while preserving scene identity across viewpoints.",
      "tasks": [
        "Motion inference",
        "Magnitude comparison",
        "Motion composition",
        "Temporal ordering",
        "Cross-view spatial inference"
      ],
      "cases": [
        "E11",
        "E12",
        "E13",
        "E14",
        "E15",
        "E16",
        "E17",
        "E18",
        "E19",
        "E20",
        "E21",
        "E22",
        "E23",
        "E24",
        "E25",
        "E26",
        "E27"
      ]
    },
    "l3": {
      "title": "L3: Long-horizon transition integration",
      "count": "22,922",
      "description": "Compose successive transitions over complete camera trajectories to recover global trajectory properties and key locations.",
      "tasks": [
        "Path length",
        "Endpoint displacement",
        "Trajectory shape",
        "Turning intervals",
        "Revisited locations & reverse paths"
      ],
      "cases": [
        "E29",
        "E30",
        "E31",
        "E32",
        "E33",
        "E34",
        "E35"
      ]
    }
  },
  "cases": {
    "E01": {
      "title": "Object toggle",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "What state change happened from Image A to Image B?",
      "options": [
        "An object was rotated.",
        "An object was opened or closed.",
        "An object was turned on or off.",
        "An object was removed from the scene."
      ],
      "answer": "C"
    },
    "E02": {
      "title": "Open / close",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "What state change happened from Image A to Image B?",
      "options": [
        "An object was rotated.",
        "An object was turned on or off.",
        "An object was opened or closed.",
        "An object was removed from the scene."
      ],
      "answer": "C"
    },
    "E03": {
      "title": "Object rotation",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "What state change happened from Image A to Image B?",
      "options": [
        "An object was turned on or off.",
        "An object was removed from the scene.",
        "An object was opened or closed.",
        "An object was rotated."
      ],
      "answer": "D"
    },
    "E04": {
      "title": "Object removal",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "What state change happened from Image A to Image B?",
      "options": [
        "An object was opened or closed.",
        "An object was removed from the scene.",
        "An object was rotated.",
        "An object was turned on or off."
      ],
      "answer": "B"
    },
    "E05": {
      "title": "Position swapping",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "What movement-related change happened from Image A to Image B?",
      "options": [
        "One object moved and changed another object's visibility or occlusion.",
        "One object moved closer to the camera.",
        "One object moved farther from the camera.",
        "Two objects swapped positions."
      ],
      "answer": "D"
    },
    "E06": {
      "title": "Dynamic occlusion",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "What movement-related change happened from Image A to Image B?",
      "options": [
        "One object moved and changed another object's visibility or occlusion.",
        "One object moved closer to the camera.",
        "Two objects swapped positions.",
        "One object moved farther from the camera."
      ],
      "answer": "A"
    },
    "E07": {
      "title": "Dynamic distance",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "What movement-related change happened from Image A to Image B?",
      "options": [
        "One object moved closer to the camera.",
        "One object moved farther from the camera.",
        "One object moved and changed another object's visibility or occlusion.",
        "Two objects swapped positions."
      ],
      "answer": "B"
    },
    "E08": {
      "title": "Action-to-image matching",
      "source": "BridgeData V2",
      "domain": "Robot trajectory",
      "question": "Starting from the leftmost view, which candidate matches a 20 cm rightward end-effector motion?",
      "options": [
        "Candidate A",
        "Candidate B",
        "Candidate C",
        "Candidate D"
      ],
      "answer": "D"
    },
    "E09": {
      "title": "Operation magnitude",
      "source": "BridgeData V2",
      "domain": "Robot trajectory",
      "question": "Given that the robot end effector moved right between Image A and Image B, approximately how far did it move in that direction?",
      "options": [
        "about 5 cm",
        "about 10 cm",
        "about 15 cm",
        "about 20 cm"
      ],
      "answer": "B"
    },
    "E10": {
      "title": "Operation ordering",
      "source": "BridgeData V2",
      "domain": "Robot trajectory",
      "question": "For the instruction 'put the silver pot in the bottom-right corner of the sink,' what is the chronological order of the four keyframes?",
      "options": [
        "A -> B -> C -> D",
        "B -> A -> C -> D",
        "A -> C -> B -> D",
        "A -> D -> C -> B"
      ],
      "answer": "C"
    },
    "E11": {
      "title": "Metric camera motion",
      "source": "ScanNet",
      "domain": "Real-world views",
      "question": "The camera mainly moves forward from Image A to Image B. Which option best estimates the translation distance?",
      "options": [
        "MoveForward 0.78 m",
        "MoveForward 0.98 m",
        "MoveForward 0.68 m",
        "MoveForward 0.88 m"
      ],
      "answer": "D"
    },
    "E12": {
      "title": "Camera rotation",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "Which single camera movement best explains the change from Image A to Image B?",
      "options": [
        "Move left by 0.6 meters",
        "Rotate left by 60 degrees",
        "Move left by 0.8 meters",
        "Rotate left by 45 degrees"
      ],
      "answer": "B"
    },
    "E13": {
      "title": "Composed camera motion",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "Which two camera movements, in order, best explain the change from Image A to Image B?",
      "options": [
        "Move forward by 0.6 meters, then rotate right by 60 degrees",
        "Rotate right by 60 degrees, then move forward by 0.6 meters",
        "Rotate right by 60 degrees, then move forward by 0.8 meters",
        "Move forward by 0.8 meters, then rotate right by 75 degrees"
      ],
      "answer": "A"
    },
    "E14": {
      "title": "Multi-frame action chain",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "Which motion chain connects the consecutive views?",
      "options": [],
      "answer": "C",
      "answerText": "C. Rotate right 60 degrees, rotate right 45 degrees, then move forward 0.4 m."
    },
    "E15": {
      "title": "Translation comparison",
      "source": "MultiScan",
      "domain": "Real-world views",
      "question": "From the same start view, which rightward endpoint has the larger translation?",
      "options": [],
      "answer": "B",
      "answerText": "B. The rightmost endpoint: 0.86 m, versus 0.44 m for the middle endpoint."
    },
    "E16": {
      "title": "Translation comparison",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "From the same start view, which backward endpoint has the larger translation?",
      "options": [],
      "answer": "D",
      "answerText": "D. The middle endpoint: 0.50 m, versus 0.25 m for the rightmost endpoint."
    },
    "E17": {
      "title": "Rotation comparison",
      "source": "ScanNet++",
      "domain": "Real-world views",
      "question": "From the same start view, which endpoint has the larger upward rotation?",
      "options": [],
      "answer": "D",
      "answerText": "D. The rightmost endpoint: 36 degrees, versus 12 degrees for the middle endpoint."
    },
    "E18": {
      "title": "Rotation comparison",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "From the same start view, which endpoint has the larger leftward rotation?",
      "options": [],
      "answer": "C",
      "answerText": "C. The middle endpoint: 60 degrees, versus 45 degrees for the rightmost endpoint."
    },
    "E19": {
      "title": "Camera temporal ordering",
      "source": "ScanNet",
      "domain": "Real-world views",
      "question": "After the fixed start view, what is the chronological order of the other three views?",
      "options": [
        "B -> A -> C",
        "A -> B -> C",
        "C -> A -> B",
        "A -> C -> B"
      ],
      "answer": "A"
    },
    "E20": {
      "title": "Camera temporal ordering",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "What is the chronological order of the four displayed camera views?",
      "options": [
        "D -> B -> C -> A",
        "A -> C -> D -> B",
        "B -> D -> C -> A",
        "D -> C -> A -> B"
      ],
      "answer": "C"
    },
    "E21": {
      "title": "Overlap-based localization",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "Using Image B's camera pose as the reference frame, where is the Lamp in Image C relative to the TV in Image A?",
      "options": [
        "back-left",
        "back-right",
        "front-left",
        "front-right"
      ],
      "answer": "A"
    },
    "E22": {
      "title": "Parallax depth",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "Using the parallax between images A and B, which is closer to the camera, vase or chair?",
      "options": [
        "cannot tell",
        "vase",
        "same depth",
        "chair"
      ],
      "answer": "B"
    },
    "E23": {
      "title": "Imagined perspective",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "Imagine standing at mirror and facing into the scene. Where is sink relative to you?",
      "options": [
        "back-left",
        "front-right",
        "front-left",
        "back-right"
      ],
      "answer": "B"
    },
    "E24": {
      "title": "Direction after motion",
      "source": "AI2-THOR",
      "domain": "Simulated",
      "question": "After rotate right 45 deg, then move forward 0.60 m from image A, where is plate relative to you?",
      "options": [
        "left",
        "back",
        "front",
        "right"
      ],
      "answer": "A"
    },
    "E25": {
      "title": "Two-view direction",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "At the viewpoint of image B, where is kettle relative to you?",
      "options": [
        "front-left",
        "front-right",
        "back-right",
        "back-left"
      ],
      "answer": "C"
    },
    "E26": {
      "title": "Visibility after motion",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "After rotate left 45 deg, then move forward 0.60 m from image A, is plant still visible? If yes, where?",
      "options": [
        "visible, center",
        "not visible",
        "visible, left",
        "visible, right"
      ],
      "answer": "B"
    },
    "E27": {
      "title": "Disappearance trend",
      "source": "ProcTHOR",
      "domain": "Simulated",
      "question": "Images A and B show the motion direction. If the same motion continues, which object leaves the view first: bowl or vase?",
      "options": [
        "both at the same time",
        "bowl",
        "neither",
        "vase"
      ],
      "answer": "D"
    },
    "E28": {
      "title": "Multi-step operation",
      "source": "BridgeData V2",
      "domain": "Robot trajectory",
      "question": "For the instruction 'put pepper in pan,' which coarse horizontal manipulation program matches the sequence?",
      "options": [
        "first moves right about 30 cm, then moves forward about 15 cm",
        "first moves right about 15 cm, then moves forward about 30 cm",
        "first moves forward about 30 cm, then moves right about 15 cm",
        "first moves right about 20 cm, then moves forward about 25 cm"
      ],
      "answer": "A"
    },
    "E29": {
      "title": "Reverse-path reasoning",
      "source": "RoomTour3D",
      "domain": "Real-world video",
      "question": "Which reverse path returns to the starting location?",
      "options": [
        "turn around, move forward, then turn right and move forward",
        "turn around, move forward, then turn left and move forward",
        "turn around, move forward",
        "move straight forward"
      ],
      "answer": "A"
    },
    "E30": {
      "title": "Metric trajectory",
      "source": "ScanNet++",
      "domain": "Real-world views",
      "question": "What are the endpoint displacement, initial-view turn, and total path length?",
      "fields": [
        {
          "label": "Endpoint displacement",
          "options": [
            "4.0 m",
            "3.5 m",
            "2.5 m",
            "3.0 m"
          ],
          "answer": "B"
        },
        {
          "label": "Initial-view turn",
          "options": [
            "0 degrees",
            "Right 30 degrees",
            "Right 90 degrees",
            "Right 60 degrees"
          ],
          "answer": "D"
        },
        {
          "label": "Path length",
          "options": [
            "4.5 m",
            "6.5 m",
            "5.5 m",
            "3.5 m"
          ],
          "answer": "A"
        }
      ],
      "answer": "B / D / A"
    },
    "E31": {
      "title": "Simulated trajectory shape",
      "source": "SIMS-V",
      "domain": "Simulated",
      "question": "Which simple path shape best describes the camera motion in this egocentric video? Reply with only one letter.",
      "options": [
        "moves forward, turns right, continues forward, then turns right and continues forward",
        "moves forward, turns left, continues forward, then turns left and continues forward",
        "moves forward, turns right, continues forward, then turns left and continues forward",
        "moves forward, turns left, continues forward, then turns right and continues forward"
      ],
      "answer": "B"
    },
    "E32": {
      "title": "Real-world trajectory shape",
      "source": "RoomTour3D",
      "domain": "Real-world video",
      "question": "Which option best describes the simple ground-plane path shape of the camera in this first-person video? Reply with only one letter.",
      "options": [
        "moves forward, turns left, then continues forward",
        "moves forward, turns left, continues forward, then turns left and continues forward",
        "moves forward, turns right, then continues forward",
        "moves mostly straight forward"
      ],
      "answer": "D"
    },
    "E33": {
      "title": "Metric trajectory",
      "source": "RoomTour3D",
      "domain": "Real-world video",
      "question": "What are the endpoint displacement, initial-view turn, and total path length?",
      "fields": [
        {
          "label": "Endpoint displacement",
          "options": [
            "6.5 m",
            "7.0 m",
            "6.0 m",
            "5.5 m"
          ],
          "answer": "A"
        },
        {
          "label": "Initial-view turn",
          "options": [
            "Right 30 degrees",
            "Right 90 degrees",
            "Right 60 degrees",
            "0 degrees"
          ],
          "answer": "A"
        },
        {
          "label": "Path length",
          "options": [
            "6.5 m",
            "8.5 m",
            "9.5 m",
            "7.5 m"
          ],
          "answer": "B"
        }
      ],
      "answer": "A / A / B"
    },
    "E34": {
      "title": "Turning-interval identification",
      "source": "RoomTour3D",
      "domain": "Real-world video",
      "question": "Which temporal quarter contains a moving turn?",
      "options": [
        "The final quarter, around frames 25-32",
        "The first quarter, around frames 1-8",
        "The third quarter, around frames 17-24",
        "The second quarter, around frames 9-16"
      ],
      "answer": "D"
    },
    "E35": {
      "title": "Revisited-location identification",
      "source": "RoomTour3D",
      "domain": "Real-world video",
      "question": "Which pair of frames shows the camera at nearly the same ground-plane location? Reply with only one letter.",
      "options": [
        "frame 1 and frame 13",
        "frame 6 and frame 32",
        "frame 1 and frame 31",
        "frame 1 and frame 30"
      ],
      "answer": "B"
    }
  }
});
