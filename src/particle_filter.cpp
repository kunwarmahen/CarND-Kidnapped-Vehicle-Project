/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
using namespace std;
const double PI = 3.1415926535897;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 50;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle particle;

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_psi(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1.0);
	
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//cout << "In prediction" << endl;
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_psi(0, std_pos[2]);

	double vdt = velocity * delta_t;
	double vyrate = velocity / yaw_rate;
	double yaw_delta = yaw_rate * delta_t;

	for (int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];

		if (fabs(yaw_rate) < 0.001) {
			particle.x += (vdt * sin(particle.theta));
			particle.y += (vdt * cos(particle.theta));
		} else {
			particle.x += vyrate * (sin(particle.theta + yaw_delta) - sin(particle.theta));
			particle.y += vyrate * (cos(particle.theta) - cos(particle.theta + yaw_delta));
			particle.theta += yaw_delta;
		}

		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_psi(gen);
		
		particles[i] = particle;

	}
	//cout << "Out prediction" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i <= observations.size(); i++) {
		LandmarkObs observered_landmark = observations[i];
		double closest = DBL_MAX;
		for (int j = 0; j <= predicted.size(); j++) {
			LandmarkObs predicted_landmark = predicted[j];
			double distance = dist(predicted_landmark.x, predicted_landmark.y, observered_landmark.x, observered_landmark.y);
			if (distance < closest) {
				closest = distance;
				observations[i].id = j;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	//cout << "In updateWeights" << endl;
	const double WEIGHT_F = (1 / (2 * PI * std_landmark[0] * std_landmark[1]));
	const double x_std_dev = pow(std_landmark[0], 2);
	const double y_std_dev = pow(std_landmark[1], 2);

	for (int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];
		particles[i].weight = 1.0;

		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs obs_landmark = observations[j];

			double tranformed_x = (obs_landmark.x * cos(particle.theta)) - (obs_landmark.y * sin(particle.theta)) + particle.x;
			double tranformed_y = (obs_landmark.x * sin(particle.theta)) + (obs_landmark.y * cos(particle.theta)) + particle.y;

			double closets = DBL_MAX;
			int closet_obs = 0;
			for (int j = 0; j <= map_landmarks.landmark_list.size(); j++) {
				double distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, tranformed_x, tranformed_y);

				if (distance <= sensor_range && distance < closets) {
					closets = distance;
					closet_obs = j;
				}
			}

			double x_diff  = pow(tranformed_x - map_landmarks.landmark_list[closet_obs].x_f, 2);
			double y_diff = pow(tranformed_y - map_landmarks.landmark_list[closet_obs].y_f, 2);
			double weight = WEIGHT_F * exp(-1 * ((x_diff / (2 * x_std_dev)) + (y_diff / (2 * y_std_dev))));
			particles[i].weight *= weight;
		}
	}
	//cout << "Out updateWeights" << endl;
}

//void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
//	std::vector<LandmarkObs> observations, Map map_landmarks) {
//	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
//	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//	//   The following is a good resource for the theory:
//	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//	//   and the following is a good resource for the actual equation to implement (look at equation 
//	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
//	//   for the fact that the map's y-axis actually points downwards.)
//	//   http://planning.cs.uiuc.edu/node99.html
//
//	//cout << "In updateWeights" << endl;
//	const double WEIGHT_F = (1 / (2 * PI * std_landmark[0] * std_landmark[1]));
//	const double x_std_dev = pow(std_landmark[0], 2);
//	const double y_std_dev = pow(std_landmark[1], 2);
//
//	for (int i = 0; i < num_particles; i++) {
//		Particle particle = particles[i];
//		particles[i].weight = 1.0;
//		std::vector<LandmarkObs> landmarks_within_range;
//		for (int j = 0; j <= map_landmarks.landmark_list.size(); j++) {
//			double distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particle.x, particle.y);
//
//			if (distance <= sensor_range) {
//				LandmarkObs landmarkObs;
//				landmarkObs.x = map_landmarks.landmark_list[j].x_f;
//				landmarkObs.y = map_landmarks.landmark_list[j].y_f;
//				landmarks_within_range.push_back(landmarkObs);
//			}
//		}
//
//		std::vector<LandmarkObs> observed_landmarks;
//		for (int j = 0; j < observations.size(); j++) {
//			LandmarkObs obs_landmark = observations[j];
//
//			double tranformed_x = (obs_landmark.x * cos(particle.theta)) - (obs_landmark.y * sin(particle.theta)) + particle.x;
//			double tranformed_y = (obs_landmark.x * sin(particle.theta)) + (obs_landmark.y * cos(particle.theta)) + particle.y;
//			obs_landmark.x = tranformed_x;
//			obs_landmark.y = tranformed_y;
//			observed_landmarks.push_back(obs_landmark);
//		}
//
//		dataAssociation(landmarks_within_range, observed_landmarks);
//
//		for (int j = 0; j < observed_landmarks.size(); j++) {
//			LandmarkObs observed_landmark = observed_landmarks[j];
//			double x_diff = pow(observed_landmark.x - landmarks_within_range[observed_landmark.id].x, 2);
//			double y_diff = pow(observed_landmark.y - landmarks_within_range[observed_landmark.id].y, 2);
//
//			double weight = WEIGHT_F * exp(-1 * ((x_diff / (2 * x_std_dev)) + (y_diff / (2 * y_std_dev))));
//			particles[i].weight *= weight;
//		}
//
//	}
//	//cout << "Out updateWeights" << endl;
//}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//cout << "In resample" << endl;
	weights.clear();
	std::random_device rd;
	std::default_random_engine gen(rd());

	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	vector<Particle> resampled_particles;
	resampled_particles.clear();

	std::discrete_distribution<int> d(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		Particle particle = particles[d(gen)];
		resampled_particles.push_back(particle);
	}

	particles = resampled_particles;
	//cout << "Out resample" << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}