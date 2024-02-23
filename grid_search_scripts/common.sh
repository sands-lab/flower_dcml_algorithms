local_to_global_epochs_mapping() {
    case "$1" in
        1) global_epochs=120 ;;
        2) global_epochs=60 ;;
        4) global_epochs=32 ;;
        *) global_epochs=0 ;;  # Default value if input is not 1, 2, or 4
    esac
}
