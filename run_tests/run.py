import argparse

from run_tests.send_mail import send_email
from run_tests.execute_script import run_initial_script


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an initial script and send the result via email")

    # Add the arguments for email and password
    parser.add_argument("--to_email", "-t", help="Recipient's email address")
    parser.add_argument("--from_email", "-f", help="Sender's email address")
    parser.add_argument("--smtp_password", "-p",
                        help="Password for the sender's email address (consider using environment variables for security)")
    parser.add_argument("--script_path", '-s', help="Path to the initial script you want to run")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Define the parameters
    script_path = args.script_path  # Path to your initial script
    subject = args.script_path
    to_email = args.to_email
    from_email = args.from_email
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = args.from_email
    smtp_password = args.smtp_password

    # Run the script
    script_output = run_initial_script(script_path)

    # Send the email with the results
    send_email(subject, script_output, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password)


if __name__ == "__main__":
    main()
