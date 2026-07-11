"""Web dashboard: type a task for the robot in a chat box, and see the
stereo camera views (with the VLM's bounding box overlaid on whatever it
last identified) and the perceived cube XYZ positions update live.

Only understands "pick up the {blue,red} cube" style instructions for now
(see `reasoning.instruction_parsing`) -- this is a UI shell around the
existing perceive -> ground -> act pieces, not a new reasoning capability.

Run with CoppeliaSim running, the full scene already built
(`python -m simulation.build_scene`), and Ollama running locally:
    python -m dashboard.app
"""

import os
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from perception.stereo_localization import CAMERA_NAMES, locate_cubes
from reasoning.instruction_parsing import extract_target_color
from reasoning.scene_recognition import capture_image
from reasoning.vlm_pick_and_place import locate_and_move
from simulation.connection import connect

REFRESH_INTERVAL_SECONDS = 2.0

_UNRECOGNIZED_COLOR_REPLY = (
    'Não entendi qual cubo você quer -- mencione a cor ("azul"/"blue" ou "vermelho"/"red").'
)


def _cube_positions(sim):
    positions = locate_cubes(sim)
    return {
        name: (None if position is None else [round(float(v), 4) for v in position])
        for name, position in positions.items()
    }


def _annotated(image, detection):
    """Build the (image, annotations) pair gr.AnnotatedImage expects. Only
    the most recently identified object is annotated -- there's one shared
    VLM detection per chat turn, not a running object tracker."""
    if detection is None or not detection.get("found"):
        return image, []
    x_min, y_min, x_max, y_max = detection["bbox"]
    box = (int(x_min), int(y_min), int(x_max), int(y_max))
    return image, [(box, detection["label"])]


def build_app():
    sim = connect()
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(1.0)

    # action.pick_and_place.Arm drives the sim with sim.setStepping(True) for
    # the whole duration of a move, stepping it many times in sequence; the
    # periodic camera-refresh timer must not call into the same ZMQ
    # connection concurrently with that, or with another chat submission,
    # since RemoteAPIClient isn't documented as thread-safe for concurrent
    # calls from multiple Gradio event-handler threads.
    sim_lock = threading.Lock()

    def refresh(detection=None):
        with sim_lock:
            left = capture_image(sim, CAMERA_NAMES[0])
            right = capture_image(sim, CAMERA_NAMES[1])
            values = _cube_positions(sim)
        return _annotated(left, detection), _annotated(right, detection), values

    def send_message(message, history):
        history = history + [{"role": "user", "content": message}]

        color = extract_target_color(message)
        if color is None:
            history.append({"role": "assistant", "content": _UNRECOGNIZED_COLOR_REPLY})
            left_view, right_view, values = refresh()
            return history, "", left_view, right_view, values

        target_label = f"the {color} cube"
        detection = None
        try:
            with sim_lock:
                result = locate_and_move(sim, target_label)
            detection = result["detection"]
            reply = (
                f"Peguei {result['cube_name']} em "
                f"{[round(p, 3) for p in result['pick_position']]} e coloquei em "
                f"{[round(p, 3) for p in result['place_position']]}."
            )
        except RuntimeError as exc:
            reply = f"Não consegui completar a tarefa: {exc}"

        history.append({"role": "assistant", "content": reply})
        left_view, right_view, values = refresh(detection)
        return history, "", left_view, right_view, values

    with gr.Blocks(title="Semantic Robot Reasoning") as demo:
        gr.Markdown(
            "# Semantic Robot Reasoning\n"
            'Digite uma instrução (ex: "pegue o cubo azul") para o robô executar.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Instruções")
                message_box = gr.Textbox(placeholder="ex: pegue o cubo vermelho", show_label=False)
                send_button = gr.Button("Enviar", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    left_view = gr.AnnotatedImage(label="left_camera")
                    right_view = gr.AnnotatedImage(label="right_camera")
                values_view = gr.JSON(label="Posições XYZ (perception.stereo_localization)")

        outputs = [chatbot, message_box, left_view, right_view, values_view]
        send_button.click(send_message, inputs=[message_box, chatbot], outputs=outputs)
        message_box.submit(send_message, inputs=[message_box, chatbot], outputs=outputs)

        timer = gr.Timer(REFRESH_INTERVAL_SECONDS)
        timer.tick(lambda: refresh(), outputs=[left_view, right_view, values_view])

    return demo


def main():
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
