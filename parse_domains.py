#!/usr/bin/env python3
"""
Domain Parser Script
Extracts domains from various sources in the network-traffic-project folder
"""

import csv
import re
import json
from pathlib import Path
from collections import defaultdict

def parse_csv_domains(csv_path):
    """Parse domains from CSV files with domain,label format"""
    domains = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'domain' in row:
                    domain = row['domain'].strip()
                    label = int(row.get('label', 0))
                    if domain:
                        domains.append({'domain': domain, 'label': label})
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return domains

def parse_log_domains(log_path):
    """Parse domains from DNS log files"""
    domains = []
    domain_pattern = re.compile(r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}')

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                matches = domain_pattern.findall(line)
                for domain in matches:
                    domains.append({'domain': domain, 'label': None})
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return domains

def main():
    # Path to network-traffic-project
    base_path = Path('/home/koshiq/network-traffic-project')

    all_domains = []
    domain_stats = defaultdict(int)

    print("=" * 70)
    print("DOMAIN PARSER - Network Traffic Project")
    print("=" * 70)

    # Parse CSV files
    csv_files = [
        base_path / 'Data' / 'dns_training_data.csv',
        base_path / 'updated_model' / 'dns_training_data_balanced.csv'
    ]

    for csv_file in csv_files:
        if csv_file.exists():
            print(f"\n[+] Parsing: {csv_file.name}")
            domains = parse_csv_domains(csv_file)
            print(f"    Found {len(domains)} domains")
            all_domains.extend(domains)

    # Parse log files
    log_files = list((base_path / 'dns-blocker-service').glob('*.log'))

    for log_file in log_files:
        if log_file.exists():
            print(f"\n[+] Parsing: {log_file.name}")
            domains = parse_log_domains(log_file)
            print(f"    Found {len(domains)} domain mentions")
            all_domains.extend(domains)

    # Remove duplicates while preserving labels
    unique_domains = {}
    for item in all_domains:
        domain = item['domain']
        if domain not in unique_domains:
            unique_domains[domain] = item
        elif item['label'] is not None:
            # Prefer entries with labels
            unique_domains[domain] = item

    # Collect statistics
    total_domains = len(unique_domains)
    labeled_count = sum(1 for d in unique_domains.values() if d['label'] is not None)
    malicious_count = sum(1 for d in unique_domains.values() if d['label'] == 1)
    benign_count = sum(1 for d in unique_domains.values() if d['label'] == 0)

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total unique domains: {total_domains}")
    print(f"Labeled domains:      {labeled_count}")
    print(f"  - Malicious (1):    {malicious_count}")
    print(f"  - Benign (0):       {benign_count}")
    print(f"Unlabeled domains:    {total_domains - labeled_count}")

    # Save outputs
    output_dir = Path('/home/koshiq/ml-hierarchical-domain-classifier/data')
    output_dir.mkdir(exist_ok=True)

    # Save all domains with labels (CSV)
    csv_output = output_dir / 'parsed_domains.csv'
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        for domain, item in sorted(unique_domains.items()):
            label = item['label'] if item['label'] is not None else ''
            writer.writerow([domain, label])

    print(f"\n[✓] Saved CSV: {csv_output}")

    # Save domains only (text file, one per line)
    txt_output = output_dir / 'domains_list.txt'
    with open(txt_output, 'w', encoding='utf-8') as f:
        for domain in sorted(unique_domains.keys()):
            f.write(f"{domain}\n")

    print(f"[✓] Saved TXT: {txt_output}")

    # Save labeled domains separately
    labeled_output = output_dir / 'labeled_domains.csv'
    with open(labeled_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        for domain, item in sorted(unique_domains.items()):
            if item['label'] is not None:
                writer.writerow([domain, item['label']])

    print(f"[✓] Saved labeled domains: {labeled_output}")

    # Save malicious domains only
    malicious_output = output_dir / 'malicious_domains.txt'
    with open(malicious_output, 'w', encoding='utf-8') as f:
        for domain, item in sorted(unique_domains.items()):
            if item['label'] == 1:
                f.write(f"{domain}\n")

    print(f"[✓] Saved malicious list: {malicious_output}")

    # Save benign domains only
    benign_output = output_dir / 'benign_domains.txt'
    with open(benign_output, 'w', encoding='utf-8') as f:
        for domain, item in sorted(unique_domains.items()):
            if item['label'] == 0:
                f.write(f"{domain}\n")

    print(f"[✓] Saved benign list: {benign_output}")

    # Save summary JSON
    summary = {
        'total_domains': total_domains,
        'labeled_domains': labeled_count,
        'malicious_count': malicious_count,
        'benign_count': benign_count,
        'unlabeled_count': total_domains - labeled_count,
        'sources': [str(f) for f in csv_files if f.exists()],
        'output_files': {
            'all_domains_csv': str(csv_output),
            'domains_txt': str(txt_output),
            'labeled_csv': str(labeled_output),
            'malicious_txt': str(malicious_output),
            'benign_txt': str(benign_output)
        }
    }

    json_output = output_dir / 'parsing_summary.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"[✓] Saved summary: {json_output}")

    print("\n" + "=" * 70)
    print("DOMAIN PARSING COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()