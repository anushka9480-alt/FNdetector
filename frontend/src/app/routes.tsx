import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { FakeNews } from "./pages/FakeNews";
import { About } from "./pages/About";
import { Contact } from "./pages/Contact";
import { Deepfake } from "./pages/Deepfake";
import { HowItWorks } from "./pages/HowItWorks";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Home },
      { path: "fake-news", Component: FakeNews },
      { path: "about", Component: About },
      { path: "contact", Component: Contact },
      { path: "deepfake", Component: Deepfake },
      { path: "how-it-works", Component: HowItWorks },
    ],
  },
]);
