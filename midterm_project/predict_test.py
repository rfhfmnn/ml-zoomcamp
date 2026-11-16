#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Zoomcamp — Midterm Project
Prediction Service Test Script (predict_test.py)

Tests the running prediction service by sending HTTP requests and inspecting responses.
"""

import sys
import requests

SERVER_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:9696"


def test_service():
    print(f"Connecting to prediction service at: {SERVER_URL}")

    try:
        ping_res = requests.get(f"{SERVER_URL}/ping", timeout=5)
        print(f"Healthcheck (/ping) status code: {ping_res.status_code}")
        print(f"Healthcheck response: {ping_res.json()}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to {SERVER_URL}. Is predict.py running?")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Testing match predictions:")
    print("=" * 50)

    match_1 = {
        "rival": "Racing Club",
        "condicion_independiente": "local",
        "anio": 2026
    }
    print(f"\n[Test 1] Independiente vs. {match_1['rival']} (Playing as {match_1['condicion_independiente']})")
    res_1 = requests.post(f"{SERVER_URL}/predict", json=match_1)
    print("Status Code:", res_1.status_code)
    data_1 = res_1.json()
    print("Response:")
    print(f"  Win probability: {data_1.get('win_probability'):.2%}")
    print(f"  Prediction:      {data_1.get('prediction')} (Victory: {data_1.get('victory')})")

    # 3. Test Case 2: Visitor against Boca Juniors
    match_2 = {
        "rival": "Boca Juniors",
        "condicion_independiente": "Visitante",
        "anio": 2026
    }
    print(f"\n[Test 2] Independiente vs. {match_2['rival']} (Playing as {match_2['condicion_independiente']})")
    res_2 = requests.post(f"{SERVER_URL}/predict", json=match_2)
    print("Status Code:", res_2.status_code)
    data_2 = res_2.json()
    print("Response:")
    print(f"  Win Probability: {data_2.get('win_probability'):.2%}")
    print(f"  Prediction:      {data_2.get('prediction')} (Victory: {data_2.get('victory')})")

    # 4. Test Case 3: Local against Banfield
    match_3 = {
        "rival": "Banfield",
        "condicion_independiente": "Local",
        "anio": 2026
    }
    print(f"\n[Test 3] Independiente vs. {match_3['rival']} (Playing as {match_3['condicion_independiente']})")
    res_3 = requests.post(f"{SERVER_URL}/predict", json=match_3)
    print("Status Code:", res_3.status_code)
    data_3 = res_3.json()
    print("Response:")
    print(f"  Win Probability: {data_3.get('win_probability'):.2%}")
    print(f"  Prediction:      {data_3.get('prediction')} (Victory: {data_3.get('victory')})")

    print("\n" + "=" * 50)
    print("All prediction tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_service()
