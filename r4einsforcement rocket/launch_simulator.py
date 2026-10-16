import os
import sys
import webbrowser

def launch():
    html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs", "flight_simulator.html"))
    if not os.path.exists(html_path):
        print(f"[Building Simulator] outputs/flight_simulator.html not found, generating now...")
        import build_flight_simulator
        build_flight_simulator.build_flight_simulator()

    file_url = f"file:///{html_path.replace(os.sep, '/')}"
    print("=" * 70)
    print("    LAUNCHING ROCKETLANDER-RL CINEMA FLIGHT SIMULATOR & ARENA")
    print("=" * 70)
    print(f"Opening: {file_url}")
    print("=" * 70)
    webbrowser.open(file_url)

if __name__ == "__main__":
    launch()
