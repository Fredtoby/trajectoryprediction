provider "google" {
  project = "spectral-trajectory-prediction"
  region  = "us-central1"
  zone    = "us-central1-c"
}

resource "google_compute_instance" "gpu-vm" {
  count = 1
  name = "gpu-vm"
  machine_type = "n1-standard-4" // 1 CPU 16 Gig of RAM
  tags = ["http"]
  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-1604-lts"
      size = 50 // 50 GB Storage
    }
   }
  network_interface {
    network = "default"
      access_config {
      }
  }
  guest_accelerator{
    type = "nvidia-tesla-k80" // Type of GPU attahced
    count = 1 // Num of GPU attached
  }
  scheduling{
    on_host_maintenance = "TERMINATE" // Need to terminate GPU on maintenance
  }
  metadata_startup_script = "${file("start-up-script.sh")}" // Here we will add the env setup
}
