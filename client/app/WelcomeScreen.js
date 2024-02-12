import React from "react";
import { View, Image, StyleSheet, Button } from "react-native";

const WelcomeScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Image
        source={require("./assets/background.jpg")} // Your welcome screen image
        style={StyleSheet.absoluteFill} // This makes the image fill the entire screen
      />
      <View style={styles.logoContainer}>
        <Image
          source={require("./assets/logo.png")}
          style={{ width: 200, height: 200 }}
          resizeMode="contain"
        />
        <View style={styles.buttonContainer}>
          <Button
            title="Get Started"
            onPress={() => navigation.navigate("ChatbotScreen")}
          />
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  logoContainer: {
    flex: 1,
    justifyContent: "center",
  },
  buttonContainer: {
    marginTop: 30,
  },
});

export default WelcomeScreen;
