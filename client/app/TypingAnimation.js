import React, { useEffect, useRef } from "react";
import { View, Animated, StyleSheet } from "react-native";

const TypingAnimation = () => {
  const fadeAnim1 = useRef(new Animated.Value(0)).current;
  const fadeAnim2 = useRef(new Animated.Value(0)).current;
  const fadeAnim3 = useRef(new Animated.Value(0)).current;

  const startAnimation = (animation) => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(animation, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(animation, {
          toValue: 0,
          duration: 600,
          useNativeDriver: true,
        }),
      ]),
      { iterations: -1 }
    ).start();
  };

  useEffect(() => {
    startAnimation(fadeAnim1);
    setTimeout(() => startAnimation(fadeAnim2), 200);
    setTimeout(() => startAnimation(fadeAnim3), 400);
  }, []);

  return (
    <View style={styles.dotsContainer}>
      <Animated.View style={[styles.dot, { opacity: fadeAnim1 }]} />
      <Animated.View style={[styles.dot, { opacity: fadeAnim2 }]} />
      <Animated.View style={[styles.dot, { opacity: fadeAnim3 }]} />
    </View>
  );
};

const styles = StyleSheet.create({
  dotsContainer: {
    flexDirection: "row",
    height: 20,
  },
  dot: {
    backgroundColor: "#000",
    width: 8,
    height: 8,
    borderRadius: 4,
    marginHorizontal: 2,
  },
});

export default TypingAnimation;
